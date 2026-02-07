#!/usr/bin/env python3
"""
Efficient NNUE Training using PyTorch

Architecture: HalfKP(41024) -> 256x2 -> 32 -> 32 -> 1
"""

from __future__ import annotations

import argparse
import random
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import chess
import dummy_chess
import numba
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn.functional as F
import tqdm.auto as tqdm

if TYPE_CHECKING:
    pass

# ============================================================================
# Architecture Constants (must match NNUE.hpp)
# ============================================================================

HALFKP_SIZE = 41024
FT_OUT = 256
L1_OUT = 32
L2_OUT = 32

FT_QUANT_SCALE = 127
WEIGHT_QUANT_SCALE = 64

PIECE_TO_INDEX = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
    [-1, -1],  # P, N, B, R, Q, K
]

# Piece char to (piece_type 0-5, is_white)
# P=0, N=1, B=2, R=3, Q=4, K=5
PIECE_CHAR_MAP = {
    "P": (0, True),
    "N": (1, True),
    "B": (2, True),
    "R": (3, True),
    "Q": (4, True),
    "K": (5, True),
    "p": (0, False),
    "n": (1, False),
    "b": (2, False),
    "r": (3, False),
    "q": (4, False),
    "k": (5, False),
}

# Numba-compatible constants (numpy arrays for JIT)
_PIECE_TO_INDEX = np.array(
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [-1, -1]], dtype=np.int32
)


@numba.jit(nopython=True, cache=True)
def _compute_halfkp_features(
    piece_squares: np.ndarray,
    piece_types: np.ndarray,
    piece_colors: np.ndarray,
    n_pieces: int,
    wk: int,
    bk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numba-optimized HalfKP feature computation.

    Args:
        piece_squares: Array of squares for each non-king piece
        piece_types: Array of piece types (0-4: P,N,B,R,Q)
        piece_colors: Array of colors (1=white, 0=black)
        n_pieces: Number of non-king pieces
        wk: White king square
        bk: Black king square

    Returns:
        white_feats, black_feats as int32 arrays
    """
    white_feats = np.empty(n_pieces, dtype=np.int32)
    black_feats = np.empty(n_pieces, dtype=np.int32)

    for i in range(n_pieces):
        sq = piece_squares[i]
        pt = piece_types[i]
        is_white = piece_colors[i]

        # Index into piece table: 0 if same color, 1 if opposite
        w_idx = 0 if is_white else 1
        b_idx = 1 if is_white else 0

        # Feature indices (matching _PIECE_TO_INDEX layout)
        # pt*2 + color_offset gives the piece index
        white_feats[i] = wk * 641 + (pt * 2 + w_idx) * 64 + sq + 1
        black_feats[i] = (63 - bk) * 641 + (pt * 2 + b_idx) * 64 + (63 - sq) + 1

    return white_feats, black_feats


# Thread-local storage for pre-allocated arrays (needed for num_workers > 0)
import threading

_MAX_PIECES = 32


class _ThreadLocalArrays(threading.local):
    """Thread-local pre-allocated arrays for FEN parsing."""

    def __init__(self):
        self.piece_squares = np.zeros(_MAX_PIECES, dtype=np.int32)
        self.piece_types = np.zeros(_MAX_PIECES, dtype=np.int32)
        self.piece_colors = np.zeros(_MAX_PIECES, dtype=np.int32)


_tls = _ThreadLocalArrays()


def parse_fen_fast(
    fen: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, bool]:
    """
    Fast FEN parser that fills thread-local numpy arrays.

    Returns:
        piece_squares, piece_types, piece_colors: Arrays with piece data
        n_pieces: Number of non-king pieces
        wk: white king square
        bk: black king square
        white_to_move: True if white to move
    """
    parts = fen.split(" ")
    board_str = parts[0]
    white_to_move = parts[1] == "w" if len(parts) > 1 else True

    # Get thread-local arrays
    piece_squares = _tls.piece_squares
    piece_types = _tls.piece_types
    piece_colors = _tls.piece_colors

    n_pieces = 0
    wk = -1
    bk = -1
    sq = 56  # Start at a8

    for c in board_str:
        if c == "/":
            sq -= 16  # Next rank
        elif c.isdigit():
            sq += int(c)
        elif c in PIECE_CHAR_MAP:
            pt, is_white = PIECE_CHAR_MAP[c]
            if pt == 5:  # King
                if is_white:
                    wk = sq
                else:
                    bk = sq
            else:
                piece_squares[n_pieces] = sq
                piece_types[n_pieces] = pt
                piece_colors[n_pieces] = 1 if is_white else 0
                n_pieces += 1
            sq += 1

    return piece_squares, piece_types, piece_colors, n_pieces, wk, bk, white_to_move


def get_halfkp_features(fen: str | bytes) -> tuple[list[int], list[int], int]:
    """Extract HalfKP features from FEN string or compressed FEN bytes (numba-accelerated)."""
    # Decompress if bytes
    if isinstance(fen, bytes):
        fen = dummy_chess.decompress_fen(fen)

    piece_squares, piece_types, piece_colors, n_pieces, wk, bk, white_to_move = (
        parse_fen_fast(fen)
    )

    if n_pieces == 0:
        return [], [], 0 if white_to_move else 1

    white_feats, black_feats = _compute_halfkp_features(
        piece_squares, piece_types, piece_colors, n_pieces, wk, bk
    )

    return white_feats.tolist(), black_feats.tolist(), 0 if white_to_move else 1


def get_halfkp_features_np(fen: str | bytes) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Extract HalfKP features from FEN string or compressed FEN bytes, returning numpy arrays.

    Faster than get_halfkp_features when arrays are needed (avoids .tolist()).
    """
    # Decompress if bytes
    if isinstance(fen, bytes):
        fen = dummy_chess.decompress_fen(fen)

    piece_squares, piece_types, piece_colors, n_pieces, wk, bk, white_to_move = (
        parse_fen_fast(fen)
    )

    if n_pieces == 0:
        empty = np.array([], dtype=np.int32)
        return empty, empty, 0 if white_to_move else 1

    white_feats, black_feats = _compute_halfkp_features(
        piece_squares, piece_types, piece_colors, n_pieces, wk, bk
    )

    return white_feats.copy(), black_feats.copy(), 0 if white_to_move else 1


# ============================================================================
# Model
# ============================================================================


class NNUE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = torch.nn.EmbeddingBag(HALFKP_SIZE, FT_OUT, mode="sum", sparse=True)
        self.ft_bias = torch.nn.Parameter(torch.zeros(FT_OUT))
        self.l1 = torch.nn.Linear(FT_OUT * 2, L1_OUT)
        self.l2 = torch.nn.Linear(L1_OUT, L2_OUT)
        self.out = torch.nn.Linear(L2_OUT, 1)

        torch.nn.init.normal_(self.ft.weight, std=0.01)
        for m in [self.l1, self.l2, self.out]:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            torch.nn.init.zeros_(m.bias)

    def forward(self, w_idx, w_off, b_idx, b_off, stm):
        w_ft = torch.clamp(self.ft(w_idx, w_off) + self.ft_bias, 0, 1)
        b_ft = torch.clamp(self.ft(b_idx, b_off) + self.ft_bias, 0, 1)
        ft = torch.where(
            stm.unsqueeze(1) == 0,
            torch.cat([w_ft, b_ft], 1),
            torch.cat([b_ft, w_ft], 1),
        )
        x = torch.clamp(self.l1(ft), 0, 1)
        x = torch.clamp(self.l2(x), 0, 1)
        return self.out(x)


# ============================================================================
# Dataset
# ============================================================================


class DataSource:
    """
    A data source with a LazyFrame and optional weight for proportional sampling.

    Args:
        path: Path to CSV file
        weight: Relative weight for sampling (default 1.0)
        name: Optional name for logging
    """

    def __init__(self, path: str, weight: float = 1.0, name: str | None = None):
        self.path = path
        self.weight = weight
        self.name = name or Path(path).stem
        self._lf: pl.LazyFrame | None = None
        self._len: int | None = None

    @property
    def lf(self) -> pl.LazyFrame:
        if self._lf is None:
            self._lf = pl.scan_csv(self.path)
        return self._lf

    def __len__(self) -> int:
        if self._len is None:
            self._len = self.lf.select(pl.len()).collect().item()
        return self._len

    def __repr__(self) -> str:
        return f"DataSource({self.name}, weight={self.weight}, len={len(self)})"


class ProportionalDataset(torch.utils.data.IterableDataset):
    """
    Dataset that samples from multiple sources with configurable proportions.

    Each epoch samples from sources according to their weights. Sources are
    shuffled independently and sampled proportionally.

    Args:
        sources: List of DataSource objects or (path, weight) tuples
        split: Which split to use ('train', 'val', 'test', or None for all)
        val_ratio: Fraction for validation set (default 0.05)
        test_ratio: Fraction for test set (default 0.05)
        seed: Random seed for shuffling and sampling
        epoch_size: Total samples per epoch (None = sum of all source lengths)

    Example:
        dataset = ProportionalDataset([
            DataSource("data/evals.csv.gz", weight=0.7),
            DataSource("data/puzzles.csv.gz", weight=0.2),
            DataSource("data/endgames.csv.gz", weight=0.1),
        ], split="train")

        # Or with tuples:
        dataset = ProportionalDataset([
            ("data/evals.csv.gz", 0.7),
            ("data/puzzles.csv.gz", 0.2),
            ("data/endgames.csv.gz", 0.1),
        ], split="train")
    """

    def __init__(
        self,
        sources: list[DataSource | tuple[str, float]],
        split: str | None = None,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 42,
        epoch_size: int | None = None,
    ):
        # Normalize sources
        self.sources = []
        for src in sources:
            if isinstance(src, DataSource):
                self.sources.append(src)
            else:
                path, weight = src
                self.sources.append(DataSource(path, weight))

        self.split = split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.epoch_size = epoch_size
        self._len: int | None = None

        # Normalize weights to sum to 1
        total_weight = sum(s.weight for s in self.sources)
        self._normalized_weights = [s.weight / total_weight for s in self.sources]

    def __len__(self) -> int:
        if self._len is None:
            if self.epoch_size is not None:
                total = self.epoch_size
            else:
                total = sum(len(s) for s in self.sources)

            # Adjust for split
            if self.split == "train":
                self._len = int(total * (1 - self.val_ratio - self.test_ratio))
            elif self.split == "val":
                self._len = int(total * self.val_ratio)
            elif self.split == "test":
                self._len = int(total * self.test_ratio)
            else:
                self._len = total

        return self._len  # type: ignore[return-value]

    def _get_split_ratio(self) -> tuple[float, float]:
        """Get (start_ratio, end_ratio) for current split."""
        if self.split == "train":
            return 0.0, 1.0 - self.val_ratio - self.test_ratio
        elif self.split == "val":
            return 1.0 - self.val_ratio - self.test_ratio, 1.0 - self.test_ratio
        elif self.split == "test":
            return 1.0 - self.test_ratio, 1.0
        else:
            return 0.0, 1.0

    def __iter__(self):
        rng = random.Random(self.seed)
        start_ratio, end_ratio = self._get_split_ratio()

        # Load and shuffle each source, then take the split portion
        source_iters = []
        for src in self.sources:
            # Shuffle deterministically using hash
            df = (
                src.lf.with_columns(
                    (pl.col("fen").hash(seed=self.seed) % (2**32)).alias("_shuffle_key")
                )
                .sort("_shuffle_key")
                .drop("_shuffle_key")
                .collect()
            )

            # Take split portion
            n = len(df)
            start_idx = int(n * start_ratio)
            end_idx = int(n * end_ratio)
            split_df = df.slice(start_idx, end_idx - start_idx)

            source_iters.append(iter(split_df.iter_rows(named=True)))

        # Sample proportionally from sources
        samples_yielded = 0
        target_samples = len(self)

        while samples_yielded < target_samples:
            # Choose source based on weights
            src_idx = rng.choices(range(len(self.sources)), self._normalized_weights)[0]

            try:
                row = next(source_iters[src_idx])
                w, b, stm = get_halfkp_features(row["fen"])
                yield w, b, stm, float(row["score"])
                samples_yielded += 1
            except StopIteration:
                # Source exhausted, reload and reshuffle
                src = self.sources[src_idx]
                df = (
                    src.lf.with_columns(
                        (
                            pl.col("fen").hash(seed=self.seed + samples_yielded)
                            % (2**32)
                        ).alias("_shuffle_key")
                    )
                    .sort("_shuffle_key")
                    .drop("_shuffle_key")
                    .collect()
                )

                n = len(df)
                start_idx = int(n * start_ratio)
                end_idx = int(n * end_ratio)
                split_df = df.slice(start_idx, end_idx - start_idx)

                source_iters[src_idx] = iter(split_df.iter_rows(named=True))
            except Exception:
                continue


class LazyFrameDataset(torch.utils.data.IterableDataset):
    """
    Dataset that accepts one or more polars LazyFrames.

    Supports train/val/test splitting and multiple data sources.

    Args:
        sources: LazyFrame or list of LazyFrames with 'fen' and 'score' columns
        split: Which split to use ('train', 'val', 'test', or None for all)
        val_ratio: Fraction for validation set (default 0.05)
        test_ratio: Fraction for test set (default 0.05)
        seed: Random seed for splitting
        chunk_size: Batch size for iterating over LazyFrames

    Example:
        # Single source
        lf = pl.scan_csv("data/evals.csv.gz")
        train_ds = LazyFrameDataset(lf, split="train")
        val_ds = LazyFrameDataset(lf, split="val")

        # Multiple sources
        lf1 = pl.scan_csv("data/puzzles.csv.gz")
        lf2 = pl.scan_csv("data/evals.csv.gz")
        dataset = LazyFrameDataset([lf1, lf2], split="train")
    """

    def __init__(
        self,
        sources: pl.LazyFrame | list[pl.LazyFrame],
        split: str | None = None,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 42,
        chunk_size: int = 50000,
    ):
        # Normalize to list
        if isinstance(sources, pl.LazyFrame):
            sources = [sources]
        self.sources = sources
        self.split = split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.chunk_size = chunk_size
        self._len: int | None = None

    def __len__(self) -> int:
        """Count total rows across all sources (cached)."""
        if self._len is None:
            total = 0
            for lf in self.sources:
                total += lf.select(pl.len()).collect().item()

            # Adjust for split
            if self.split == "train":
                self._len = int(total * (1 - self.val_ratio - self.test_ratio))
            elif self.split == "val":
                self._len = int(total * self.val_ratio)
            elif self.split == "test":
                self._len = int(total * self.test_ratio)
            else:
                self._len = total

        return self._len  # type: ignore[return-value]

    def _get_split_indices(self, n: int) -> tuple[int, int]:
        """Get start and end indices for the current split."""
        n_test = int(n * self.test_ratio)
        n_val = int(n * self.val_ratio)
        n_train = n - n_val - n_test

        if self.split == "train":
            return 0, n_train
        elif self.split == "val":
            return n_train, n_train + n_val
        elif self.split == "test":
            return n_train + n_val, n
        else:
            return 0, n

    def __iter__(self):
        # Concatenate all sources
        if len(self.sources) == 1:
            combined = self.sources[0]
        else:
            combined = pl.concat(self.sources)

        # Get total count for splitting
        total = combined.select(pl.len()).collect().item()
        start_idx, end_idx = self._get_split_indices(total)

        # Add row numbers and filter to split
        combined = combined.with_row_index("_row_idx")

        # Apply deterministic shuffle using hash of fen + seed
        combined = combined.with_columns(
            (pl.col("fen").hash(seed=self.seed) % (2**32)).alias("_shuffle_key")
        ).sort("_shuffle_key")

        # Filter to split range
        combined = combined.filter(
            (pl.col("_row_idx") >= start_idx) & (pl.col("_row_idx") < end_idx)
        ).drop("_row_idx", "_shuffle_key")

        # Iterate in chunks
        for batch in combined.collect().iter_slices(self.chunk_size):
            for row in batch.iter_rows(named=True):
                try:
                    w, b, stm = get_halfkp_features(row["fen"])
                    yield w, b, stm, float(row["score"])
                except Exception:
                    continue


def collate_sparse(batch):
    """Collate sparse HalfKP features into batched tensors."""
    # Pre-compute sizes for numpy pre-allocation
    n = len(batch)
    w_lens = [len(b[0]) for b in batch]
    b_lens = [len(b[1]) for b in batch]
    w_total = sum(w_lens)
    b_total = sum(b_lens)

    # Pre-allocate numpy arrays
    w_all = np.empty(w_total, dtype=np.int64)
    b_all = np.empty(b_total, dtype=np.int64)
    w_off = np.empty(n, dtype=np.int64)
    b_off = np.empty(n, dtype=np.int64)
    stm_arr = np.empty(n, dtype=np.int64)
    score_arr = np.empty(n, dtype=np.float32)

    # Fill arrays
    w_pos, b_pos = 0, 0
    for i, (w, b, stm, score) in enumerate(batch):
        w_len = w_lens[i]
        b_len = b_lens[i]

        w_off[i] = w_pos
        b_off[i] = b_pos

        w_all[w_pos : w_pos + w_len] = w
        b_all[b_pos : b_pos + b_len] = b

        w_pos += w_len
        b_pos += b_len

        stm_arr[i] = stm
        score_arr[i] = score

    return (
        torch.from_numpy(w_all),
        torch.from_numpy(w_off),
        torch.from_numpy(b_all),
        torch.from_numpy(b_off),
        torch.from_numpy(stm_arr),
        torch.from_numpy(score_arr).unsqueeze(1),
    )


# ============================================================================
# Training
# ============================================================================


class Tracker:
    """
    Training metrics tracker with epoch averaging.

    Tracks metrics by split (e.g., 'train', 'val') and name. Uses defaultdict(list)
    for consistent tracking logic across all metrics.

    Usage:
        tracker = Tracker()
        pbar = tqdm(total=total_batches * epochs, desc="Training")
        for epoch in range(epochs):
            for batch in train_loader:
                loss = train_step(batch)
                tracker.track("train", "loss", loss.item())
                pbar.set_postfix(**tracker.postfix)
                pbar.update(1)
            # Run validation
            for batch in val_loader:
                tracker.track("val", "loss", compute_val_loss(batch))
                pbar.update(1)
            tracker.submit_epoch()
            tracker.track_epoch("epoch", epoch + 1)
        pbar.close()
    """

    def __init__(self):
        from collections import defaultdict

        self._current: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._history: list[dict[str, dict[str, float]]] = []
        self._epoch = 0
        self._epoch_metrics: dict[str, str] = {}

    def track(self, split: str, name: str, value: float) -> None:
        """Track a metric value for the current epoch."""
        self._current[split][name].append(value)

    def track_epoch(self, name: str, value: float | int | str) -> None:
        """Track an epoch-level metric (shown in postfix, not averaged)."""
        self._epoch_metrics[name] = f"{value}"

    @property
    def postfix(self) -> dict[str, str]:
        """Get current aggregate values for tqdm postfix."""
        result = dict(self._epoch_metrics)
        for split, metrics in self._current.items():
            for name, values in metrics.items():
                if values:
                    avg = sum(values) / len(values)
                    result[f"{split}_{name}"] = f"{avg:.4f}"
        return result

    def submit_epoch(self) -> dict[str, dict[str, float]]:
        """
        Finalize current epoch: average all tracked metrics.

        Returns:
            Dict of {split: {name: avg_value}} for all tracked metrics
        """
        self._epoch += 1

        # Average all tracked metrics
        metrics: dict[str, dict[str, float]] = {"epoch": {"n": float(self._epoch)}}
        for split, split_metrics in self._current.items():
            metrics[split] = {}
            for name, values in split_metrics.items():
                metrics[split][name] = sum(values) / len(values) if values else 0.0

        # Store in history and reset
        self._history.append(metrics)
        self._current.clear()

        return metrics

    def __getitem__(self, key: str) -> dict[str, float]:
        """Get last epoch's metrics for a split (e.g., tracker['train'])."""
        if self._history:
            return self._history[-1].get(key, {})
        return {}

    @property
    def history(self) -> list[dict[str, dict[str, float]]]:
        """Get full training history."""
        return self._history

    @property
    def epoch(self) -> int:
        """Get current epoch number."""
        return self._epoch

    def best(self, split: str, name: str) -> float:
        """Get best (minimum) value for a metric across all epochs."""
        values = [
            m[split][name]
            for m in self._history
            if split in m and name in m[split] and not np.isnan(m[split][name])
        ]
        return min(values) if values else float("inf")


def evaluate(model, loader, device):
    model.eval()
    total_loss, n = 0, 0
    with torch.no_grad():
        for batch in loader:
            w_idx, w_off, b_idx, b_off, stm, target = [x.to(device) for x in batch]
            pred = model(w_idx, w_off, b_idx, b_off, stm)
            total_loss += F.mse_loss(pred, target).item()
            n += 1
    return total_loss / max(n, 1)


def train(
    train_dataset,
    val_dataset,
    output: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[NNUE, Tracker]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    is_iterable = isinstance(
        train_dataset, torch.utils.data.IterableDataset
    ) or not hasattr(train_dataset, "__getitem__")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_sparse,
        num_workers=0,
        shuffle=not is_iterable,
    )
    n_train_batches = (len(train_dataset) + batch_size - 1) // batch_size
    val_loader = (
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_sparse,
            num_workers=0,
        )
        if val_dataset is not None
        else None
    )

    model = NNUE().to(device)
    tracker = Tracker()

    # Use SparseAdam for sparse embedding, AdamW for dense layers
    sparse_params = [model.ft.weight]
    dense_params = [p for n, p in model.named_parameters() if "ft.weight" not in n]

    sparse_optimizer = torch.optim.SparseAdam(sparse_params, lr=lr)
    dense_optimizer = torch.optim.AdamW(dense_params, lr=lr, weight_decay=1e-4)

    # Use CosineAnnealingLR which steps per epoch (no fixed steps_per_epoch needed)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        dense_optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    best_val = float("inf")

    # Calculate total batches for single progress bar across all epochs
    n_val_batches = (
        (len(val_dataset) + batch_size - 1) // batch_size
        if val_dataset is not None
        else 0
    )
    batches_per_epoch = n_train_batches + n_val_batches
    total_batches = batches_per_epoch * epochs

    pbar = tqdm.tqdm(total=total_batches, desc="Training")
    tracker.track_epoch("epoch", f"1/{epochs}")

    for epoch in range(epochs):
        model.train()
        tracker.track_epoch("epoch", f"{epoch + 1}/{epochs}")

        # Training
        for batch in train_loader:
            w_idx, w_off, b_idx, b_off, stm, target = [x.to(device) for x in batch]
            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast("cuda"):
                    loss = F.mse_loss(model(w_idx, w_off, b_idx, b_off, stm), target)
                scaler.scale(loss).backward()
                scaler.step(sparse_optimizer)
                scaler.step(dense_optimizer)
                scaler.update()
            else:
                loss = F.mse_loss(model(w_idx, w_off, b_idx, b_off, stm), target)
                loss.backward()
                sparse_optimizer.step()
                dense_optimizer.step()

            tracker.track("train", "loss", loss.item())
            pbar.set_postfix(**tracker.postfix)
            pbar.update(1)

        # Validation
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    w_idx, w_off, b_idx, b_off, stm, target = [
                        x.to(device) for x in batch
                    ]
                    pred = model(w_idx, w_off, b_idx, b_off, stm)
                    val_loss = F.mse_loss(pred, target).item()
                    tracker.track("val", "loss", val_loss)
                    pbar.set_postfix(**tracker.postfix)
                    pbar.update(1)

        scheduler.step()
        tracker.submit_epoch()

        val_loss = tracker["val"].get("loss", float("inf"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output.replace(".nnue", ".pt"))
            export_nnue(model, output)

    pbar.close()

    return model, tracker


# ============================================================================
# Export
# ============================================================================


def export_nnue(model: NNUE, path: str):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x7AF32F20))
        f.write(struct.pack("<I", 0))
        arch = b"Features=HalfKP(Friend)[41024->256x2]->[32->32]->1"
        f.write(struct.pack("<I", len(arch)))
        f.write(arch)
        f.write(struct.pack("<I", 0x5D69D5B9))

        # Feature transformer
        bias = (
            (model.ft_bias.detach().cpu().numpy() * FT_QUANT_SCALE)
            .clip(-32768, 32767)
            .astype(np.int16)
        )
        f.write(bias.tobytes())
        weight = (
            (model.ft.weight.detach().cpu().numpy().T * FT_QUANT_SCALE)
            .clip(-32768, 32767)
            .astype(np.int16)
        )
        f.write(weight.T.tobytes())

        f.write(struct.pack("<I", 0))

        # Hidden layers
        for layer, scale in [
            (model.l1, FT_QUANT_SCALE * WEIGHT_QUANT_SCALE),
            (model.l2, WEIGHT_QUANT_SCALE * WEIGHT_QUANT_SCALE),
        ]:
            f.write(
                (layer.bias.detach().cpu().numpy() * scale).astype(np.int32).tobytes()
            )
            f.write(
                (layer.weight.detach().cpu().numpy().T * WEIGHT_QUANT_SCALE)
                .clip(-128, 127)
                .astype(np.int8)
                .tobytes()
            )

        # Output
        f.write(
            (model.out.bias.detach().cpu().numpy() * WEIGHT_QUANT_SCALE**2)
            .astype(np.int32)
            .tobytes()
        )
        f.write(
            (model.out.weight.detach().cpu().numpy().flatten() * WEIGHT_QUANT_SCALE)
            .clip(-128, 127)
            .astype(np.int8)
            .tobytes()
        )

    print(f"Exported: {path}")


# ============================================================================
# Evaluation
# ============================================================================


def _get_model_device(model: torch.nn.Module) -> torch.device:
    """Get the device of a model's parameters."""
    return next(model.parameters()).device


def evaluate_fen(
    fen: str, model: NNUE | None = None, model_path: str | None = None
) -> float:
    """
    Evaluate a FEN position using the NNUE model.

    Args:
        fen: FEN string of the position to evaluate
        model: Pre-loaded NNUE model (optional)
        model_path: Path to .pt model file (used if model is None)

    Returns:
        Evaluation score in centipawns from the side to move's perspective
    """
    if model is None:
        if model_path is None:
            model_path = str(Path(__file__).parent / "network.pt")
        model = NNUE()
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        model.eval()

    device = _get_model_device(model)
    w_feats, b_feats, stm = get_halfkp_features(fen)

    with torch.no_grad():
        w_idx = torch.tensor(w_feats, dtype=torch.long, device=device)
        w_off = torch.tensor([0], dtype=torch.long, device=device)
        b_idx = torch.tensor(b_feats, dtype=torch.long, device=device)
        b_off = torch.tensor([0], dtype=torch.long, device=device)
        stm_t = torch.tensor([stm], dtype=torch.long, device=device)

        score = model(w_idx, w_off, b_idx, b_off, stm_t).item()

    return score


def evaluate_fens(
    fens: list[str | bytes], model: NNUE | None = None, model_path: str | None = None
) -> list[float]:
    """
    Evaluate multiple FEN positions using the NNUE model (batched).

    Args:
        fens: List of FEN strings or compressed FEN bytes to evaluate
        model: Pre-loaded NNUE model (optional)
        model_path: Path to .pt model file (used if model is None)

    Returns:
        List of evaluation scores in centipawns from each side to move's perspective
    """
    if model is None:
        if model_path is None:
            model_path = str(Path(__file__).parent / "network.pt")
        model = NNUE()
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        model.eval()

    device = _get_model_device(model)

    # Batch decompress any compressed FENs
    fen_strs: list[str] = []
    compressed_indices: list[int] = []
    compressed_fens: list[bytes] = []

    for i, fen in enumerate(fens):
        if isinstance(fen, bytes):
            compressed_indices.append(i)
            compressed_fens.append(fen)
            fen_strs.append("")  # placeholder
        else:
            fen_strs.append(fen)

    if compressed_fens:
        decompressed = dummy_chess.decompress_fens_batch(compressed_fens)
        for i, idx in enumerate(compressed_indices):
            fen_strs[idx] = decompressed[i]

    # Extract features for all positions
    batch = [get_halfkp_features(fen) for fen in fen_strs]

    # Collate into batched tensors
    w_all, b_all, w_off, b_off = [], [], [0], [0]
    stm_list = []
    for w, b, stm in batch:
        w_all.extend(w)
        b_all.extend(b)
        w_off.append(len(w_all))
        b_off.append(len(b_all))
        stm_list.append(stm)

    with torch.no_grad():
        w_idx = torch.tensor(w_all, dtype=torch.long, device=device)
        w_off_t = torch.tensor(w_off[:-1], dtype=torch.long, device=device)
        b_idx = torch.tensor(b_all, dtype=torch.long, device=device)
        b_off_t = torch.tensor(b_off[:-1], dtype=torch.long, device=device)
        stm_t = torch.tensor(stm_list, dtype=torch.long, device=device)

        scores = model(w_idx, w_off_t, b_idx, b_off_t, stm_t).squeeze(-1).tolist()

    return scores


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import polars as pl

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data",
        nargs="+",
        help="One or more CSV/parquet files (will be combined for training)",
    )
    parser.add_argument("--output", default="network.nnue")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load all data sources as LazyFrames
    sources = []
    for path in args.data:
        if not Path(path).exists():
            print(f"Error: {path} not found")
            sys.exit(1)
        if path.endswith(".parquet"):
            sources.append(pl.scan_parquet(path))
        else:
            sources.append(pl.scan_csv(path))
        print(f"Added: {path}")

    # Create train/val datasets with splitting
    train_dataset = LazyFrameDataset(
        sources,
        split="train",
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    val_dataset = LazyFrameDataset(
        sources,
        split="val",
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"Train: {len(train_dataset)} rows, Val: {len(val_dataset)} rows")

    train(
        train_dataset, val_dataset, args.output, args.epochs, args.batch_size, args.lr
    )
