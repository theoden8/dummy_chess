#!/usr/bin/env python3
"""
Efficient NNUE Training using PyTorch

Architecture: HalfKP(41024) -> 256x2 -> 32 -> 32 -> 1
"""

import __future__

import argparse
import dataclasses
import pathlib
import random
import struct
import sys
import threading
import typing

import chess
import dummy_chess
import numba
import numpy as np
import polars as pl
import pyarrow.parquet
import torch
import torch.nn.functional
import tqdm.auto as tqdm

# ============================================================================
# Architecture Constants (must match NNUE.hpp)
# ============================================================================

# Feature sizes for different architectures
HALFKP_SIZE = dummy_chess.HALFKP_SIZE  # 41025 (64 king squares * 640 + 1)
HALFKAV2_SIZE = dummy_chess.HALFKAV2_SIZE  # 7693 (12 king buckets * 640 + 1)

# Common layer sizes
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
    """
    NNUE network supporting both HalfKP and HalfKAv2 architectures.

    Args:
        arch: "halfkp" (default) or "halfkav2"
    """

    def __init__(self, arch: str = "halfkp"):
        super().__init__()
        self.arch = arch

        if arch == "halfkp":
            ft_size = HALFKP_SIZE
        elif arch == "halfkav2":
            ft_size = HALFKAV2_SIZE
        else:
            raise ValueError(
                f"Unknown architecture: {arch}. Use 'halfkp' or 'halfkav2'"
            )

        self.ft_size = ft_size
        self.ft = torch.nn.EmbeddingBag(ft_size, FT_OUT, mode="sum", sparse=True)
        self.ft_bias = torch.nn.Parameter(torch.zeros(FT_OUT))
        self.l1 = torch.nn.Linear(FT_OUT * 2, L1_OUT)
        self.l2 = torch.nn.Linear(L1_OUT, L2_OUT)
        self.out = torch.nn.Linear(L2_OUT, 1)

        torch.nn.init.normal_(self.ft.weight, std=0.01)
        for m in [self.l1, self.l2, self.out]:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            torch.nn.init.zeros_(m.bias)

    def forward(self, w_idx, w_off, b_idx, b_off, stm):
        # ReLU during training; clamp to [0,127] happens in quantized C++ inference
        w_ft = torch.relu(self.ft(w_idx, w_off) + self.ft_bias)
        b_ft = torch.relu(self.ft(b_idx, b_off) + self.ft_bias)
        ft = torch.where(
            stm.unsqueeze(1) == 0,
            torch.cat([w_ft, b_ft], 1),
            torch.cat([b_ft, w_ft], 1),
        )
        x = torch.relu(self.l1(ft))
        x = torch.relu(self.l2(x))
        return self.out(x)


# ============================================================================
# Dataset
# ============================================================================


@dataclasses.dataclass
class SplitConfig:
    """
    Configuration for deterministic train/val/test splitting.

    Data should be pre-shuffled during preprocessing. This config only controls
    which portion of the data to use for each split (no runtime shuffling).

    Example:
        split_cfg = SplitConfig(val_ratio=0.1, test_ratio=0.1)
        train_ds = LazyFrameDataset(sources, split="train", split_config=split_cfg)
        val_ds = LazyFrameDataset(sources, split="val", split_config=split_cfg)
    """

    val_ratio: float = 0.05
    test_ratio: float = 0.05

    def get_split_indices(self, n: int, split: str | None) -> tuple[int, int]:
        """Get start and end indices for a split."""
        n_test = int(n * self.test_ratio)
        n_val = int(n * self.val_ratio)
        n_train = n - n_val - n_test

        if split == "train":
            return 0, n_train
        elif split == "val":
            return n_train, n_train + n_val
        elif split == "test":
            return n_train + n_val, n
        else:
            return 0, n

    def get_split_len(self, total: int, split: str | None) -> int:
        """Get the length of a split."""
        start, end = self.get_split_indices(total, split)
        return end - start


class LazyFrameDataset(torch.utils.data.IterableDataset):
    """
    Streaming dataset for large parquet files.

    Streams data directly from parquet without loading into memory.
    Data should be pre-shuffled during preprocessing.

    Args:
        sources: LazyFrame, or list of LazyFrames, or list of (LazyFrame, weight) tuples
        split: Which split to use ('train', 'val', 'test', or None for all)
        split_config: SplitConfig for splitting (uses row ranges, no shuffle)
        chunk_size: Number of rows to process at a time
        flip_augment: If True, yield both original and flipped positions (2x data)
        arch: Architecture for feature extraction ('halfkp' or 'halfkav2')

    Example:
        split_cfg = SplitConfig(val_ratio=0.05, test_ratio=0.05)

        # Single source (pre-shuffled)
        lf = pl.scan_parquet("data/evals.parquet")
        train_ds = LazyFrameDataset(lf, split="train", split_config=split_cfg)

        # Multiple sources with weights (70% evals, 20% puzzles, 10% endgames)
        train_ds = LazyFrameDataset([
            (pl.scan_parquet("data/evals.parquet"), 0.7),
            (pl.scan_parquet("data/puzzles.parquet"), 0.2),
            (pl.scan_parquet("data/endgames.parquet"), 0.1),
        ], split="train", split_config=split_cfg)
    """

    def __init__(
        self,
        sources: pl.LazyFrame | list[pl.LazyFrame] | list[tuple[pl.LazyFrame, float]],
        split: str | None = None,
        split_config: SplitConfig | None = None,
        chunk_size: int = 50000,
        flip_augment: bool = False,
        arch: str = "halfkp",
    ):
        # Normalize to list of (LazyFrame, weight)
        if isinstance(sources, pl.LazyFrame):
            self.sources = [(sources, 1.0)]
        elif isinstance(sources, list) and len(sources) > 0:
            if isinstance(sources[0], tuple):
                self.sources = list(sources)  # Already (lf, weight) tuples
            else:
                self.sources = [(lf, 1.0) for lf in sources]  # Equal weights
        else:
            self.sources = []

        self.split = split
        self.split_config = split_config or SplitConfig()
        self.chunk_size = chunk_size
        self.flip_augment = flip_augment
        self.arch = arch
        self._len: int | None = None
        self._source_lens: list[int] | None = None

        # Select feature extractor based on architecture
        if arch == "halfkp":
            self._get_features = dummy_chess.get_halfkp_features_batch
        elif arch == "halfkav2":
            self._get_features = dummy_chess.get_halfkav2_features_batch
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def _get_source_lens(self) -> list[int]:
        """Get row counts for each source (cached)."""
        if self._source_lens is None:
            self._source_lens = []
            for lf, _ in self.sources:
                n = lf.select(pl.len()).collect(engine="streaming").item()
                self._source_lens.append(n)
        return self._source_lens

    def _get_weighted_counts(self) -> list[int]:
        """Calculate sample counts per source respecting weights with strict ratio."""
        source_lens = self._get_source_lens()
        weights = [w for _, w in self.sources]
        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]

        split_lens = [
            self.split_config.get_split_len(n, self.split) for n in source_lens
        ]

        # Find max total that respects strict ratio (limited by smallest source)
        # For each source: max_total = available / weight
        max_totals = [
            avail / w if w > 0 else float("inf")
            for avail, w in zip(split_lens, norm_weights)
        ]
        max_total = min(max_totals)

        # Calculate actual counts maintaining strict ratio
        return [int(max_total * w) for w in norm_weights]

    def __len__(self) -> int:
        """Count samples in this split (cached)."""
        if self._len is None:
            self._len = sum(self._get_weighted_counts())
            if self.flip_augment:
                self._len *= 2
        return self._len

    def __iter__(self):
        source_lens = self._get_source_lens()
        weights = [w for _, w in self.sources]
        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]

        # Get weighted counts (strict ratio)
        actual_counts = self._get_weighted_counts()

        # Create iterators for each source's split (limited to actual_count)
        source_iters = []
        for i, (lf, _) in enumerate(self.sources):
            n = source_lens[i]
            start_idx, end_idx = self.split_config.get_split_indices(n, self.split)

            # Limit to actual_count samples
            end_idx = min(end_idx, start_idx + actual_counts[i])

            if start_idx == 0:
                split_lf = lf.head(end_idx)
            else:
                split_size = end_idx - start_idx
                split_lf = lf.slice(start_idx, split_size)

            source_iters.append(split_lf.collect_batches(chunk_size=self.chunk_size))

        # Interleave sources round-robin style
        source_buffers: list[list] = [[] for _ in self.sources]
        source_exhausted = [False] * len(self.sources)

        def refill_buffer(idx: int) -> bool:
            """Try to refill buffer for source idx. Returns False if exhausted."""
            if source_exhausted[idx]:
                return False
            try:
                batch = next(source_iters[idx])
                fens = batch["fen"].to_list()
                scores = batch["score"].to_list()

                w_idx, w_off, b_idx, b_off, stm = self._get_features(fens, False)

                n_samples = len(fens)
                for i in range(n_samples):
                    w_start = w_off[i]
                    w_end = w_off[i + 1] if i + 1 < n_samples else len(w_idx)
                    b_start = b_off[i]
                    b_end = b_off[i + 1] if i + 1 < n_samples else len(b_idx)

                    source_buffers[idx].append(
                        (
                            w_idx[w_start:w_end].copy(),
                            b_idx[b_start:b_end].copy(),
                            int(stm[i]),
                            float(scores[i]),
                        )
                    )

                # Also add flipped if augmenting
                if self.flip_augment:
                    w_idx_f, w_off_f, b_idx_f, b_off_f, stm_f = self._get_features(
                        fens, True
                    )
                    for i in range(n_samples):
                        w_start = w_off_f[i]
                        w_end = w_off_f[i + 1] if i + 1 < n_samples else len(w_idx_f)
                        b_start = b_off_f[i]
                        b_end = b_off_f[i + 1] if i + 1 < n_samples else len(b_idx_f)

                        source_buffers[idx].append(
                            (
                                w_idx_f[w_start:w_end].copy(),
                                b_idx_f[b_start:b_end].copy(),
                                int(stm_f[i]),
                                -float(scores[i]),
                            )
                        )

                return True
            except StopIteration:
                source_exhausted[idx] = True
                return False

        # Yield samples interleaved by weight
        rng = random.Random(42)  # Deterministic interleaving
        while True:
            # Check if all sources exhausted
            all_empty = all(
                len(buf) == 0 and source_exhausted[i]
                for i, buf in enumerate(source_buffers)
            )
            if all_empty:
                break

            # Pick source based on weights (only from non-exhausted sources with data)
            available = [
                i
                for i in range(len(self.sources))
                if len(source_buffers[i]) > 0 or not source_exhausted[i]
            ]
            if not available:
                break

            # Weighted choice among available sources
            avail_weights = [norm_weights[i] for i in available]
            total_avail = sum(avail_weights)
            avail_weights = [w / total_avail for w in avail_weights]

            src_idx = rng.choices(available, avail_weights)[0]

            # Refill if needed
            if len(source_buffers[src_idx]) == 0:
                if not refill_buffer(src_idx):
                    continue

            # Yield sample
            if source_buffers[src_idx]:
                yield source_buffers[src_idx].pop(0)


class ShuffledParquetDataset(torch.utils.data.IterableDataset):
    """
    Shuffled streaming dataset using multi-row-group interleaving.

    Maintains multiple "active" row groups and samples randomly from them,
    providing good shuffling without loading entire dataset into memory.

    Args:
        paths: List of parquet file paths (or list of (path, weight) tuples)
        split: Which split to use ('train', 'val', 'test', or None for all)
        split_config: SplitConfig for splitting
        num_active_groups: Number of row groups to keep active (more = better shuffle, more RAM)
        buffer_per_group: Samples to buffer from each active group
        flip_augment: If True, yield both original and flipped positions
        seed: Random seed for shuffling (changes each epoch if None)
        arch: Architecture for feature extraction ('halfkp' or 'halfkav2')

    Memory usage: ~num_active_groups * buffer_per_group * 50 bytes
    Default: 8 * 4096 * 50 = ~1.6 MB
    """

    def __init__(
        self,
        paths: list[str] | list[tuple[str, float]],
        split: str | None = None,
        split_config: SplitConfig | None = None,
        num_active_groups: int = 8,
        buffer_per_group: int = 4096,
        flip_augment: bool = False,
        seed: int | None = None,
        arch: str = "halfkp",
    ):
        # Normalize to list of (path, weight)
        if isinstance(paths[0], str):
            self.paths = [(p, 1.0) for p in paths]
        else:
            self.paths = list(paths)

        self.split = split
        self.split_config = split_config or SplitConfig()
        self.num_active_groups = num_active_groups
        self.buffer_per_group = buffer_per_group
        self.flip_augment = flip_augment
        self.seed = seed
        self.arch = arch
        self._epoch = 0

        # Select feature extractor based on architecture
        if arch == "halfkp":
            self._get_features = dummy_chess.get_halfkp_features_batch
        elif arch == "halfkav2":
            self._get_features = dummy_chess.get_halfkav2_features_batch
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # Cache metadata
        self._metadata: list[tuple[str, int, list[tuple[int, int]]]] | None = None
        self._len: int | None = None

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling."""
        self._epoch = epoch

    def _get_metadata(self) -> list[tuple[str, int, list[tuple[int, int]], float]]:
        """Get row group metadata for all files. Returns [(path, total_rows, [(rg_start, rg_len), ...], weight)]."""
        if self._metadata is not None:
            return self._metadata

        self._metadata = []
        for path, weight in self.paths:
            pf = pyarrow.parquet.ParquetFile(path)
            meta = pf.metadata

            row_groups = []
            offset = 0
            for i in range(meta.num_row_groups):
                rg_len = meta.row_group(i).num_rows
                row_groups.append((offset, rg_len))
                offset += rg_len

            self._metadata.append((path, offset, row_groups, weight))

        return self._metadata

    def _get_split_row_groups(self) -> list[tuple[str, int, int, float]]:
        """Get row groups that fall within our split. Returns [(path, rg_start, rg_len, weight), ...]."""
        metadata = self._get_metadata()
        result = []

        for path, total_rows, row_groups, weight in metadata:
            split_start, split_end = self.split_config.get_split_indices(
                total_rows, self.split
            )

            for rg_start, rg_len in row_groups:
                rg_end = rg_start + rg_len

                # Check if row group overlaps with split
                if rg_end <= split_start or rg_start >= split_end:
                    continue

                # Clip to split boundaries
                actual_start = max(rg_start, split_start)
                actual_end = min(rg_end, split_end)
                actual_len = actual_end - actual_start

                if actual_len > 0:
                    result.append((path, actual_start, actual_len, weight))

        return result

    def __len__(self) -> int:
        if self._len is None:
            row_groups = self._get_split_row_groups()
            self._len = sum(rg[2] for rg in row_groups)
            if self.flip_augment:
                self._len *= 2
        return self._len

    def __iter__(self):
        row_groups = self._get_split_row_groups()
        if not row_groups:
            return

        # Shuffle row groups for this epoch
        rng = random.Random(self.seed if self.seed is not None else self._epoch)
        row_groups = list(row_groups)
        rng.shuffle(row_groups)

        # Active group state: {idx: (path, pf, current_offset, end_offset, buffer, weight)}
        active: dict[int, tuple] = {}
        next_group_idx = 0

        def fill_active():
            """Fill active slots up to num_active_groups."""
            nonlocal next_group_idx
            while len(active) < self.num_active_groups and next_group_idx < len(
                row_groups
            ):
                path, start, length, weight = row_groups[next_group_idx]
                pf = pyarrow.parquet.ParquetFile(path)
                active[next_group_idx] = {
                    "path": path,
                    "pf": pf,
                    "offset": start,
                    "end": start + length,
                    "buffer": [],
                    "weight": weight,
                }
                next_group_idx += 1

        def refill_buffer(idx: int) -> bool:
            """Refill buffer for active group. Returns False if exhausted."""
            state = active[idx]
            if state["offset"] >= state["end"]:
                return False

            # Read a chunk
            read_len = min(self.buffer_per_group, state["end"] - state["offset"])

            # Use polars for efficient reading with offset
            df = (
                pl.scan_parquet(state["path"])
                .slice(state["offset"], read_len)
                .collect()
            )
            state["offset"] += read_len

            fens = df["fen"].to_list()
            scores = df["score"].to_list()

            # Extract features
            w_idx, w_off, b_idx, b_off, stm = self._get_features(fens, False)

            n_samples = len(fens)
            for i in range(n_samples):
                w_start = w_off[i]
                w_end = w_off[i + 1] if i + 1 < n_samples else len(w_idx)
                b_start = b_off[i]
                b_end = b_off[i + 1] if i + 1 < n_samples else len(b_idx)

                state["buffer"].append(
                    (
                        w_idx[w_start:w_end].copy(),
                        b_idx[b_start:b_end].copy(),
                        int(stm[i]),
                        float(scores[i]),
                    )
                )

            # Also add flipped if augmenting
            if self.flip_augment:
                w_idx_f, w_off_f, b_idx_f, b_off_f, stm_f = self._get_features(
                    fens, True
                )
                for i in range(n_samples):
                    w_start = w_off_f[i]
                    w_end = w_off_f[i + 1] if i + 1 < n_samples else len(w_idx_f)
                    b_start = b_off_f[i]
                    b_end = b_off_f[i + 1] if i + 1 < n_samples else len(b_idx_f)

                    state["buffer"].append(
                        (
                            w_idx_f[w_start:w_end].copy(),
                            b_idx_f[b_start:b_end].copy(),
                            int(stm_f[i]),
                            -float(scores[i]),
                        )
                    )

            # Shuffle buffer
            rng.shuffle(state["buffer"])
            return True

        # Initial fill
        fill_active()
        for idx in list(active.keys()):
            refill_buffer(idx)

        # Yield samples
        while active:
            # Pick random active group (weighted)
            indices = list(active.keys())
            weights = [active[i]["weight"] for i in indices]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]

            idx = rng.choices(indices, weights)[0]
            state = active[idx]

            # Get sample from buffer
            if state["buffer"]:
                yield state["buffer"].pop()

            # Refill if empty
            if not state["buffer"]:
                if not refill_buffer(idx):
                    # This group is exhausted, remove and fill replacement
                    del active[idx]
                    fill_active()
                    for new_idx in list(active.keys()):
                        if not active[new_idx]["buffer"]:
                            if not refill_buffer(new_idx):
                                del active[new_idx]


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
            total_loss += torch.nn.functional.mse_loss(pred, target).item()
            n += 1
    return total_loss / max(n, 1)


def train(
    train_dataset,
    val_dataset,
    output: str,
    epochs: int,
    batch_size: int,
    lr: float,
    tracker: Tracker | None = None,
    num_workers: int = 0,
    arch: str = "halfkp",
) -> tuple[NNUE, Tracker]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: {arch}")

    is_iterable = isinstance(
        train_dataset, torch.utils.data.IterableDataset
    ) or not hasattr(train_dataset, "__getitem__")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_sparse,
        num_workers=num_workers,
        shuffle=not is_iterable,
    )
    n_train_batches = (len(train_dataset) + batch_size - 1) // batch_size
    val_loader = (
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_sparse,
            num_workers=num_workers,
        )
        if val_dataset is not None
        else None
    )
    n_val_batches = (
        (len(val_dataset) + batch_size - 1) // batch_size
        if val_dataset is not None
        else 0
    )

    model = NNUE(arch=arch).to(device)
    if tracker is None:
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
                    loss = torch.nn.functional.mse_loss(
                        model(w_idx, w_off, b_idx, b_off, stm), target
                    )
                scaler.scale(loss).backward()
                scaler.step(sparse_optimizer)
                scaler.step(dense_optimizer)
                scaler.update()
            else:
                loss = torch.nn.functional.mse_loss(
                    model(w_idx, w_off, b_idx, b_off, stm), target
                )
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
                    val_loss = torch.nn.functional.mse_loss(pred, target).item()
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
            model_path = str(pathlib.Path(__file__).parent / "network.pt")
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
            model_path = str(pathlib.Path(__file__).parent / "network.pt")
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

    parser = argparse.ArgumentParser(description="Train NNUE network.")
    parser.add_argument(
        "data",
        nargs="+",
        help="One or more parquet files",
    )
    parser.add_argument("--output", default="network.nnue")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument(
        "--flip-augment",
        action="store_true",
        help="Augment data by including flipped positions (2x data, zero mean scores)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Use shuffled multi-row-group sampling (recommended for training)",
    )
    parser.add_argument(
        "--num-active-groups",
        type=int,
        default=8,
        help="Number of row groups to sample from simultaneously (more = better shuffle)",
    )
    parser.add_argument(
        "--buffer-per-group",
        type=int,
        default=4096,
        help="Samples to buffer per active row group",
    )
    parser.add_argument(
        "--arch",
        choices=["halfkp", "halfkav2"],
        default="halfkp",
        help="NNUE architecture: halfkp (41K features) or halfkav2 (7.7K features, king buckets)",
    )
    args = parser.parse_args()

    # Validate paths
    for path in args.data:
        if not pathlib.Path(path).exists():
            print(f"Error: {path} not found")
            sys.exit(1)
        print(f"Added: {path}")

    # Create split config
    split_config = SplitConfig(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    if args.shuffle:
        # Use shuffled dataset (recommended for training)
        train_dataset = ShuffledParquetDataset(
            args.data,
            split="train",
            split_config=split_config,
            num_active_groups=args.num_active_groups,
            buffer_per_group=args.buffer_per_group,
            flip_augment=args.flip_augment,
            arch=args.arch,
        )
        val_dataset = ShuffledParquetDataset(
            args.data,
            split="val",
            split_config=split_config,
            num_active_groups=4,  # Fewer for validation
            buffer_per_group=args.buffer_per_group,
            flip_augment=False,
            arch=args.arch,
        )
    else:
        # Use sequential dataset (for compatibility)
        sources: list[pl.LazyFrame] = []
        for path in args.data:
            if path.endswith(".parquet"):
                sources.append(pl.scan_parquet(path))
            else:
                sources.append(pl.scan_csv(path))

        train_dataset = LazyFrameDataset(
            sources,
            split="train",
            split_config=split_config,
            flip_augment=args.flip_augment,
            arch=args.arch,
        )
        val_dataset = LazyFrameDataset(
            sources,
            split="val",
            split_config=split_config,
            flip_augment=False,
            arch=args.arch,
        )

    print(f"Train: {len(train_dataset)} rows, Val: {len(val_dataset)} rows")

    train(
        train_dataset,
        val_dataset,
        args.output,
        args.epochs,
        args.batch_size,
        args.lr,
        arch=args.arch,
    )
