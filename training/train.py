#!/usr/bin/env python3
"""
Efficient NNUE Training using PyTorch

Architecture: HalfKP(41024) -> 256x2 -> 32 -> 32 -> 1

IMPORTANT: Run with 8GB memory limit to prevent OOM:
    ulimit -v 8388608 && uv run python train.py ...
"""

import __future__

import argparse
import collections
import concurrent.futures
import dataclasses
import pathlib
import queue
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

import _preprocess

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

        # Output scaling factor - model outputs are scaled to centipawn range
        # This ensures gradients flow properly with sigmoid-based loss
        self.output_scale = SIGMOID_SCALE

        torch.nn.init.normal_(self.ft.weight, std=0.01)
        for m in [self.l1, self.l2, self.out]:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            torch.nn.init.zeros_(m.bias)

    @staticmethod
    def _clipped_relu(x: torch.Tensor, upper: float) -> torch.Tensor:
        """Leaky ClippedReLU: linear in [0, upper], small slope outside.

        Matches hard clamp(0, upper) at inference but preserves gradient
        during training so neurons above the upper bound can recover.
        Uses slope 0.01 below 0 and above upper (like leaky ReLU).
        """
        _LEAK = 0.01
        return torch.where(
            x <= 0,
            _LEAK * x,
            torch.where(x >= upper, upper + _LEAK * (x - upper), x),
        )

    def forward(self, w_idx, w_off, b_idx, b_off, stm):
        # Leaky ClippedReLU bounds matching quantized C++ inference:
        # FT output: int16 -> clamp(0, 127) -> int8, i.e. float [0, 1.0]
        # Hidden layers: int32 -> /64 -> clamp(0, 127), i.e. float [0, 127/64]
        _FT_CLAMP = 127.0 / FT_QUANT_SCALE  # 1.0
        _HL_CLAMP = 127.0 / WEIGHT_QUANT_SCALE  # 127/64 ≈ 1.984

        w_ft = self._clipped_relu(self.ft(w_idx, w_off) + self.ft_bias, _FT_CLAMP)
        b_ft = self._clipped_relu(self.ft(b_idx, b_off) + self.ft_bias, _FT_CLAMP)
        ft = torch.where(
            stm.unsqueeze(1) == 0,
            torch.cat([w_ft, b_ft], 1),
            torch.cat([b_ft, w_ft], 1),
        )
        x = self._clipped_relu(self.l1(ft), _HL_CLAMP)
        x = self._clipped_relu(self.l2(x), _HL_CLAMP)
        out = self.out(x) * self.output_scale
        # Negate output for black's perspective (stm=1) since scores are from white's POV
        return torch.where(stm.unsqueeze(1) == 0, out, -out)


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


class ParquetDataset(torch.utils.data.IterableDataset):
    """
    Unified high-performance dataset for parquet files.

    Combines features of BatchedParquetDataset and PrefetchedDataset into one class.
    Uses pyarrow row-group streaming to handle arbitrarily large files.

    Args:
        paths: List of parquet file paths
        batch_size: Samples per batch. If None, yields individual samples (slower).
        split: Which split to use ('train', 'val', 'test', or None for all)
        split_config: SplitConfig for train/val/test splitting
        arch: Architecture for feature extraction ('halfkp' or 'halfkav2')
        flip_augment: If True, also yield flipped positions with negated scores (2x data)
        prefetch: Number of batches to prefetch in background thread (0 = disabled)

    Example:
        # Fast batched training (recommended)
        train_ds = ParquetDataset(
            ["evals.parquet", "endgames.parquet"],
            batch_size=8192,
            split="train",
            prefetch=4,
        )

        # Individual samples (for compatibility, slower)
        train_ds = ParquetDataset(
            ["evals.parquet"],
            batch_size=None,
            split="train",
        )
    """

    def __init__(
        self,
        paths: list[str],
        batch_size: int | None = 8192,
        split: str | None = None,
        split_config: SplitConfig | None = None,
        arch: str = "halfkp",
        flip_augment: bool = False,
        prefetch: int = 2,
    ):
        self.paths = paths if isinstance(paths, list) else [paths]
        self.batch_size = batch_size
        self.split = split
        self.split_config = split_config or SplitConfig()
        self.arch = arch
        self.flip_augment = flip_augment
        self.prefetch = prefetch
        self._len: int | None = None
        self._metadata: list[tuple[str, int]] | None = None

        if arch not in ("halfkp", "halfkav2"):
            raise ValueError(f"Unknown architecture: {arch}")

    def _get_metadata(self) -> list[tuple[str, int]]:
        """Get row counts from parquet metadata (no data loading)."""
        if self._metadata is None:
            self._metadata = []
            for path in self.paths:
                pf = pyarrow.parquet.ParquetFile(path)
                self._metadata.append((path, pf.metadata.num_rows))
        return self._metadata

    def __len__(self) -> int:
        if self._len is None:
            total = 0
            for path, n in self._get_metadata():
                split_len = self.split_config.get_split_len(n, self.split)
                total += split_len
            if self.flip_augment:
                total *= 2
            self._len = total
        return self._len

    def _build_chunk_list(self) -> list[tuple[str, int, int, int]]:
        """Build list of (path, rg_idx, slice_start, slice_end) for all row groups."""
        chunks = []
        for path, total_rows in self._get_metadata():
            start, end = self.split_config.get_split_indices(total_rows, self.split)
            pf = pyarrow.parquet.ParquetFile(path)
            meta = pf.metadata

            row_offset = 0
            for rg_idx in range(meta.num_row_groups):
                rg_rows = meta.row_group(rg_idx).num_rows
                rg_start = row_offset
                rg_end = row_offset + rg_rows

                if rg_end <= start:
                    row_offset = rg_end
                    continue
                if rg_start >= end:
                    break

                slice_start = max(0, start - rg_start)
                slice_end = min(rg_rows, end - rg_start)
                chunks.append((path, rg_idx, slice_start, slice_end))
                row_offset = rg_end

        return chunks

    def _read_chunk(self, path: str, rg_idx: int, slice_start: int, slice_end: int):
        """Read a single row group chunk and return Arrow buffers + scores.

        Returns (fen_col, data_ptr, offsets_ptr, n, scores) where the pointers
        are raw addresses for direct C++ access via the bulk API.
        """
        pf = pyarrow.parquet.ParquetFile(path)

        # Read columns separately — reading them together causes a penalty
        # in PyArrow's to_numpy() due to mixed binary/numeric types
        fen_table = pf.read_row_group(rg_idx, columns=["fen"])
        score_table = pf.read_row_group(rg_idx, columns=["score"])
        rg_rows = fen_table.num_rows

        if slice_start > 0 or slice_end < rg_rows:
            fen_table = fen_table.slice(slice_start, slice_end - slice_start)
            score_table = score_table.slice(slice_start, slice_end - slice_start)

        # Cast to large_binary for contiguous int64 offsets buffer
        fen_col = fen_table.column("fen")
        if fen_col.type != pyarrow.large_binary():
            fen_col = fen_col.cast(pyarrow.large_binary())
        if fen_col.num_chunks > 1:
            fen_col = fen_col.combine_chunks()
        else:
            fen_col = fen_col.chunk(0)

        fen_buffers = fen_col.buffers()
        scores = score_table.column("score").to_numpy().astype(np.float32)

        # Return raw buffer addresses — fen_col must stay alive (caller holds ref)
        return (
            fen_col,
            fen_buffers[2].address,
            fen_buffers[1].address,
            len(fen_col),
            scores,
        )

    def _get_torch_fn(self):
        """Return the torch-native extraction function for current arch."""
        if self.arch == "halfkp":
            return _preprocess.get_halfkp_features_torch
        elif self.arch == "halfkav2":
            return _preprocess.get_halfkav2_features_torch
        else:
            raise ValueError(f"Unknown architecture: {self.arch}")

    def _iter_batched(self):
        """Yield pre-collated tensor batches (fast path), shuffled across all sources."""
        chunks = self._build_chunk_list()
        random.shuffle(chunks)
        batch_size = self.batch_size or 50000
        torch_fn = self._get_torch_fn()

        for path, rg_idx, slice_start, slice_end in chunks:
            fen_col, data_ptr, offsets_ptr, n, scores = self._read_chunk(
                path, rg_idx, slice_start, slice_end
            )

            for start in range(0, n, batch_size):
                count = min(batch_size, n - start)
                batch_scores = scores[start : start + count]

                w_idx, w_off, b_idx, b_off, stm = torch_fn(
                    data_ptr, offsets_ptr, start, count
                )
                yield (
                    w_idx,
                    w_off,
                    b_idx,
                    b_off,
                    stm,
                    torch.from_numpy(batch_scores).unsqueeze(1),
                )

                if self.flip_augment:
                    w_f, wo_f, b_f, bo_f, stm_f = torch_fn(
                        data_ptr, offsets_ptr, start, count, True
                    )
                    yield (
                        w_f,
                        wo_f,
                        b_f,
                        bo_f,
                        stm_f,
                        torch.from_numpy(-batch_scores).unsqueeze(1),
                    )

    def _iter_samples(self):
        """Yield individual samples (slow path, for compatibility), shuffled across all sources."""
        chunks = self._build_chunk_list()
        random.shuffle(chunks)
        torch_fn = self._get_torch_fn()

        for path, rg_idx, slice_start, slice_end in chunks:
            fen_col, data_ptr, offsets_ptr, n, scores = self._read_chunk(
                path, rg_idx, slice_start, slice_end
            )

            # Use large batches for extraction, yield individual samples
            batch_size = 50000
            for start in range(0, n, batch_size):
                count = min(batch_size, n - start)
                w_idx, w_off, b_idx, b_off, stm = torch_fn(
                    data_ptr, offsets_ptr, start, count
                )
                batch_scores = scores[start : start + count]

                for i in range(count):
                    w_start = w_off[i]
                    w_end = w_off[i + 1] if i + 1 < count else len(w_idx)
                    b_start = b_off[i]
                    b_end = b_off[i + 1] if i + 1 < count else len(b_idx)

                    yield (
                        w_idx[w_start:w_end].clone(),
                        b_idx[b_start:b_end].clone(),
                        int(stm[i]),
                        float(batch_scores[i]),
                    )

                    if self.flip_augment:
                        w_f, wo_f, b_f, bo_f, stm_f = torch_fn(
                            data_ptr, offsets_ptr, start, count, True
                        )
                        yield (
                            w_f[w_start:w_end].clone(),
                            b_f[b_start:b_end].clone(),
                            int(stm_f[i]),
                            -float(batch_scores[i]),
                        )

    def _iter_chunk_subset(self, chunk_subset):
        """Yield batches from a subset of row-group chunks.

        Same logic as _iter_batched but only for the given chunks.
        Used by multi-producer prefetching so each thread works on
        independent row groups.
        """
        batch_size = self.batch_size or 50000
        torch_fn = self._get_torch_fn()

        for path, rg_idx, slice_start, slice_end in chunk_subset:
            fen_col, data_ptr, offsets_ptr, n, scores = self._read_chunk(
                path, rg_idx, slice_start, slice_end
            )

            for start in range(0, n, batch_size):
                count = min(batch_size, n - start)
                batch_scores = scores[start : start + count]

                w_idx, w_off, b_idx, b_off, stm = torch_fn(
                    data_ptr, offsets_ptr, start, count
                )
                yield (
                    w_idx,
                    w_off,
                    b_idx,
                    b_off,
                    stm,
                    torch.from_numpy(batch_scores).unsqueeze(1),
                )

                if self.flip_augment:
                    w_f, wo_f, b_f, bo_f, stm_f = torch_fn(
                        data_ptr, offsets_ptr, start, count, True
                    )
                    yield (
                        w_f,
                        wo_f,
                        b_f,
                        bo_f,
                        stm_f,
                        torch.from_numpy(-batch_scores).unsqueeze(1),
                    )

    # Number of parallel producer threads for prefetching.
    _NUM_PRODUCERS = 3

    def _iter_with_prefetch(self, base_iter):
        """Wrap iterator with multi-producer background prefetching.

        Splits row-group chunks across ``_NUM_PRODUCERS`` threads so that
        Parquet I/O and C++ feature extraction run concurrently across
        independent row groups.  All producers feed a single bounded queue
        consumed by the training loop.

        Falls back to a single producer when there are fewer chunks than
        producers (e.g. small files or validation sets).
        """
        chunks = self._build_chunk_list()
        random.shuffle(chunks)

        n_producers = min(self._NUM_PRODUCERS, len(chunks))
        if n_producers <= 1:
            # Single-producer fast path (original behaviour)
            q: queue.Queue = queue.Queue(maxsize=self.prefetch)
            sentinel = object()

            def single_producer():
                for item in base_iter:
                    q.put(item)
                q.put(sentinel)

            thread = threading.Thread(target=single_producer, daemon=True)
            thread.start()

            while True:
                item = q.get()
                if item is sentinel:
                    break
                yield item
            thread.join()
            return

        # --- Multi-producer path ---
        # Each producer puts a unique sentinel when done.  Consumer counts
        # sentinels to know when all producers have finished.
        q = queue.Queue(maxsize=self.prefetch)
        _SENTINEL = None  # sentinels are None; real items are tuples

        def producer(chunk_subset):
            for item in self._iter_chunk_subset(chunk_subset):
                q.put(item)
            q.put(_SENTINEL)

        # Round-robin distribute chunks to producers
        subsets: list[list] = [[] for _ in range(n_producers)]
        for i, chunk in enumerate(chunks):
            subsets[i % n_producers].append(chunk)

        threads = []
        for subset in subsets:
            t = threading.Thread(target=producer, args=(subset,), daemon=True)
            t.start()
            threads.append(t)

        # Consumer: drain queue until all producers have sent their sentinel
        done_count = 0
        while done_count < n_producers:
            item = q.get()
            if item is _SENTINEL:
                done_count += 1
            else:
                yield item

        for t in threads:
            t.join()

    def __iter__(self):
        if self.batch_size is not None:
            base_iter = self._iter_batched()
        else:
            base_iter = self._iter_samples()

        if self.prefetch > 0 and self.batch_size is not None:
            yield from self._iter_with_prefetch(base_iter)
        else:
            yield from base_iter

    @property
    def is_batched(self) -> bool:
        """True if dataset yields pre-batched tensors."""
        return self.batch_size is not None


def collate_sparse(batch):
    """Collate sparse HalfKP features into batched tensors."""
    n = len(batch)
    w_lens = [len(b[0]) for b in batch]
    b_lens = [len(b[1]) for b in batch]
    w_total = sum(w_lens)
    b_total = sum(b_lens)

    w_all = np.empty(w_total, dtype=np.int64)
    b_all = np.empty(b_total, dtype=np.int64)
    w_off = np.empty(n, dtype=np.int64)
    b_off = np.empty(n, dtype=np.int64)
    stm_arr = np.empty(n, dtype=np.int64)
    score_arr = np.empty(n, dtype=np.float32)

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
# GPU-side feature extraction dataset
# ============================================================================


class GpuExtractDataset:
    """
    Streaming dataset that extracts NNUE features on-device using CUDA kernels.

    Only metadata is stored at init time.  During iteration, one parquet row
    group (~100K rows, ~2-3 MB) is loaded to GPU at a time, features are
    extracted, batches are yielded, and the row group's GPU memory is freed
    before the next one is loaded.  This keeps GPU memory usage constant
    regardless of dataset size.

    Requires the ``_extract_cuda`` extension (built with nvcc).

    Args:
        paths: Parquet file paths
        batch_size: Samples per batch
        split: 'train', 'val', 'test', or None
        split_config: SplitConfig for splitting
        arch: 'halfkp' or 'halfkav2'
        flip_augment: Include flipped positions (2x data)
        device: torch device (must be CUDA)
    """

    def __init__(
        self,
        paths: list[str],
        batch_size: int = 65536,
        split: str | None = None,
        split_config: SplitConfig | None = None,
        arch: str = "halfkp",
        flip_augment: bool = False,
        device: torch.device | None = None,
    ):
        self.batch_size = batch_size
        self.arch = arch
        self.flip_augment = flip_augment
        self.device = device or torch.device("cuda")
        self.split = split
        self.split_config = split_config or SplitConfig()

        if arch not in ("halfkp", "halfkav2"):
            raise ValueError(f"Unknown architecture: {arch}")

        # Scan metadata only — no data loaded.  Build a list of
        # (path, rg_idx, slice_lo, slice_hi, n_rows) for each row group
        # that overlaps the requested split.
        self._chunks: list[tuple[str, int, int, int, int]] = []
        self.n_positions = 0

        paths = paths if isinstance(paths, list) else [paths]

        for path in paths:
            pf = pyarrow.parquet.ParquetFile(path)
            file_rows = pf.metadata.num_rows
            start, end = self.split_config.get_split_indices(file_rows, split)
            if start >= end:
                continue

            rg_row_offset = 0
            for rg_idx in range(pf.metadata.num_row_groups):
                rg_nrows = pf.metadata.row_group(rg_idx).num_rows
                rg_start = rg_row_offset
                rg_end = rg_row_offset + rg_nrows
                rg_row_offset = rg_end

                if rg_end <= start or rg_start >= end:
                    continue

                slice_lo = max(start - rg_start, 0)
                slice_hi = min(end - rg_start, rg_nrows)
                n_rows = slice_hi - slice_lo
                if n_rows > 0:
                    self._chunks.append((path, rg_idx, slice_lo, slice_hi, n_rows))
                    self.n_positions += n_rows

        if not self._chunks:
            raise ValueError("No data found for the requested split")

        print(
            f"GPU extract: {self.n_positions} positions across "
            f"{len(self._chunks)} row groups (streaming)"
        )

    def __len__(self) -> int:
        n = self.n_positions
        if self.flip_augment:
            n *= 2
        return n

    @property
    def n_batches(self) -> int:
        """Exact batch count per epoch (accounts for per-row-group tails)."""
        bs = self.batch_size
        n = sum((rows + bs - 1) // bs for _, _, _, _, rows in self._chunks)
        if self.flip_augment:
            n *= 2
        return n

    def _get_extract_fn(self):
        """Return the CUDA extraction function for current arch."""
        import _extract_cuda

        if self.arch == "halfkp":
            return _extract_cuda.extract_halfkp_gpu
        elif self.arch == "halfkav2":
            return _extract_cuda.extract_halfkav2_gpu
        else:
            raise ValueError(f"Unknown architecture: {self.arch}")

    @staticmethod
    def _load_row_group(
        path: str,
        rg_idx: int,
        slice_lo: int,
        slice_hi: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Load one row group to GPU.  Returns (data, offsets, scores, n_rows)."""
        pf = pyarrow.parquet.ParquetFile(path)
        table = pf.read_row_group(rg_idx, columns=["fen", "score"])
        rg_nrows = len(table)

        if slice_lo > 0 or slice_hi < rg_nrows:
            table = table.slice(slice_lo, slice_hi - slice_lo)

        n_rows = len(table)

        fen_col = table.column("fen")
        if fen_col.type != pyarrow.large_binary():
            fen_col = fen_col.cast(pyarrow.large_binary())
        if fen_col.num_chunks > 1:
            fen_col = fen_col.combine_chunks()
        else:
            fen_col = fen_col.chunk(0)

        bufs = fen_col.buffers()
        raw_data = np.frombuffer(bufs[2], dtype=np.uint8)
        raw_off = np.frombuffer(bufs[1], dtype=np.int64)

        # Rebase offsets to start at 0
        offsets = raw_off[: n_rows + 1].copy()
        base = offsets[0]
        data = raw_data[base : offsets[-1]].copy()
        offsets -= base

        data_gpu = torch.from_numpy(data).to(device, non_blocking=True)
        off_gpu = torch.from_numpy(offsets).to(device, non_blocking=True)

        scores_np = table.column("score").to_numpy().astype(np.float32)
        scores_gpu = (
            torch.from_numpy(scores_np).unsqueeze(1).to(device, non_blocking=True)
        )

        del table, fen_col, bufs, raw_data, raw_off, data, offsets, scores_np
        return data_gpu, off_gpu, scores_gpu, n_rows

    def iter_batches(self, shuffle: bool = True):
        """
        Yield (w_idx, w_off, b_idx, b_off, stm, target) tuples, all on GPU.

        Streams one row group at a time from disk to GPU.  A background thread
        prefetches the next row group while training runs on the current one,
        hiding I/O latency.  At most two row groups are on GPU simultaneously
        (current + prefetched).
        """
        extract_fn = self._get_extract_fn()
        bs = self.batch_size

        # Shuffle row groups within each file (preserves sequential I/O per file)
        # then shuffle the file order itself.
        if shuffle:
            # Group chunk indices by file path
            file_groups: dict[str, list[int]] = {}
            for ci, (path, _, _, _, _) in enumerate(self._chunks):
                file_groups.setdefault(path, []).append(ci)
            # Shuffle within each file, then shuffle file order
            group_keys = list(file_groups.keys())
            random.shuffle(group_keys)
            chunk_order: list[int] = []
            for key in group_keys:
                group = file_groups[key]
                random.shuffle(group)
                chunk_order.extend(group)
        else:
            chunk_order = list(range(len(self._chunks)))

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Submit first row group load
        ci = chunk_order[0]
        path, rg_idx, slice_lo, slice_hi, _ = self._chunks[ci]
        future = executor.submit(
            self._load_row_group, path, rg_idx, slice_lo, slice_hi, self.device
        )

        for i, ci in enumerate(chunk_order):
            # Wait for current row group
            data_gpu, off_gpu, scores_gpu, n_rows = future.result()

            # Prefetch next row group in background
            if i + 1 < len(chunk_order):
                next_ci = chunk_order[i + 1]
                np_, rg_, sl_, sh_, _ = self._chunks[next_ci]
                future = executor.submit(
                    self._load_row_group, np_, rg_, sl_, sh_, self.device
                )

            # Iterate over batches within this row group
            for start in range(0, n_rows, bs):
                count = min(bs, n_rows - start)
                w_idx, w_off, b_idx, b_off, stm = extract_fn(
                    data_gpu, off_gpu, start, count, False
                )
                target = scores_gpu[start : start + count]
                yield w_idx, w_off, b_idx, b_off, stm, target

                if self.flip_augment:
                    w_f, wo_f, b_f, bo_f, stm_f = extract_fn(
                        data_gpu, off_gpu, start, count, True
                    )
                    yield w_f, wo_f, b_f, bo_f, stm_f, -target

            # Free this row group's GPU memory before loading the next
            del data_gpu, off_gpu, scores_gpu

        executor.shutdown(wait=False)


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
        self._current: dict[str, dict[str, list[float]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
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


# ============================================================================
# Loss Functions
# ============================================================================

# Scaling factor for sigmoid transformation (Stockfish uses 410)
# Maps ~400cp to ~76% win probability
SIGMOID_SCALE = 400.0


def wdl_loss(
    pred: torch.Tensor, target: torch.Tensor, scale: float = SIGMOID_SCALE
) -> torch.Tensor:
    """
    WDL-style loss using sigmoid scaling.

    Converts both prediction and target to win probabilities using sigmoid,
    then computes MSE. This naturally handles the wide score range (-15000 to +15000)
    by compressing extreme values.

    Args:
        pred: Model predictions in centipawns
        target: Target scores in centipawns
        scale: Sigmoid scaling factor (default 400, Stockfish uses 410)

    Returns:
        Scalar loss tensor
    """
    pred_prob = torch.sigmoid(pred / scale)
    target_prob = torch.sigmoid(target / scale)
    return torch.nn.functional.mse_loss(pred_prob, target_prob)


def evaluate(model, loader, device, loss_fn=None):
    """Evaluate model on a data loader."""
    if loss_fn is None:
        loss_fn = wdl_loss
    model.eval()
    total_loss, n = 0, 0
    with torch.no_grad():
        for batch in loader:
            w_idx, w_off, b_idx, b_off, stm, target = [x.to(device) for x in batch]
            pred = model(w_idx, w_off, b_idx, b_off, stm)
            total_loss += loss_fn(pred, target).item()
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
    arch: str | None = None,
    quiet: bool = False,
    accum_steps: int = 1,
) -> tuple[NNUE, Tracker]:
    # Infer arch from dataset if not explicitly provided
    dataset_arch = getattr(train_dataset, "arch", None)
    if arch is None:
        arch = dataset_arch or "halfkp"
    elif dataset_arch is not None and arch != dataset_arch:
        raise ValueError(
            f"arch mismatch: train() got arch='{arch}' but dataset uses "
            f"arch='{dataset_arch}'. Remove the arch argument from train() "
            f"to use the dataset's architecture."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: {arch}")
    if accum_steps > 1:
        print(f"Gradient accumulation: {accum_steps} micro-batches per step")

    # Detect GPU extraction dataset
    gpu_extract = isinstance(train_dataset, GpuExtractDataset)
    if gpu_extract:
        print("Using GPU-side feature extraction (CUDA kernels)")

    # Check if dataset yields pre-batched tensors
    is_prebatched = (
        isinstance(train_dataset, ParquetDataset) and train_dataset.is_batched
    )

    is_iterable = isinstance(
        train_dataset, torch.utils.data.IterableDataset
    ) or not hasattr(train_dataset, "__getitem__")

    if gpu_extract:
        # GpuExtractDataset knows its exact batch count (per-row-group tails)
        n_train_batches = train_dataset.n_batches
        n_val_batches = val_dataset.n_batches if val_dataset is not None else 0
    elif is_prebatched:
        # Pre-batched dataset: no collate needed, prefetching is built-in
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=None,  # Dataset yields complete batches
        )
        n_train_batches = (len(train_dataset) + batch_size - 1) // batch_size
        val_loader = (
            torch.utils.data.DataLoader(val_dataset, batch_size=None)
            if val_dataset is not None
            else None
        )
        n_val_batches = (
            (len(val_dataset) + batch_size - 1) // batch_size
            if val_dataset is not None
            else 0
        )
    else:
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
    model.ft.sparse = (
        False  # Dense gradients — faster than SparseAdam at high batch sizes
    )
    if tracker is None:
        tracker = Tracker()

    # Single optimizer with per-group weight decay:
    # no decay on embedding (it's a lookup table), decay on dense layers
    dense_params = [p for n, p in model.named_parameters() if "ft.weight" not in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.ft.weight], "weight_decay": 0},
            {"params": dense_params, "weight_decay": 1e-4},
        ],
        lr=lr,
    )

    # Use CosineAnnealingLR which steps per epoch (no fixed steps_per_epoch needed)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    best_val = float("inf")

    # Calculate total batches for single progress bar across all epochs
    batches_per_epoch = n_train_batches + n_val_batches
    total_batches = batches_per_epoch * epochs

    pbar = tqdm.tqdm(total=total_batches, desc="Training", disable=quiet)
    tracker.track_epoch("epoch", f"1/{epochs}")

    for epoch in range(epochs):
        model.train()
        tracker.track_epoch("epoch", f"{epoch + 1}/{epochs}")

        # Training
        # Accumulate loss entirely on GPU — only sync once at epoch end.
        loss_accum = torch.zeros(1, device=device)
        loss_count = 0
        loss_scale = 1.0 / accum_steps

        optimizer.zero_grad(set_to_none=True)
        train_iter = (
            train_dataset.iter_batches(shuffle=True) if gpu_extract else train_loader
        )
        for step, batch in enumerate(train_iter):
            if gpu_extract:
                # GpuExtractDataset yields tensors already on GPU
                w_idx, w_off, b_idx, b_off, stm, target = batch
            else:
                w_idx = batch[0].to(device, non_blocking=True)
                w_off = batch[1].to(device, non_blocking=True)
                b_idx = batch[2].to(device, non_blocking=True)
                b_off = batch[3].to(device, non_blocking=True)
                stm = batch[4].to(device, non_blocking=True)
                target = batch[5].to(device, non_blocking=True)

            pred = model(w_idx, w_off, b_idx, b_off, stm)
            loss = wdl_loss(pred, target)
            # Scale loss for gradient accumulation so effective gradient
            # equals the mean over all micro-batches in an accumulation window.
            (loss * loss_scale).backward()

            loss_accum.add_(loss.detach())
            loss_count += 1
            pbar.update(1)

            if loss_count % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # Flush partial accumulation at epoch boundary
        if loss_count % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Sync once at epoch end for logging
        avg_loss = (loss_accum / loss_count).item()
        tracker.track("train", "loss", avg_loss)
        pbar.set_postfix(**tracker.postfix)

        # Validation
        val_iter = None
        if gpu_extract and val_dataset is not None:
            val_iter = val_dataset.iter_batches(shuffle=False)
        elif not gpu_extract and val_loader is not None:
            val_iter = val_loader

        if val_iter is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_iter:
                    if gpu_extract:
                        w_idx, w_off, b_idx, b_off, stm, target = batch
                    else:
                        w_idx = batch[0].to(device, non_blocking=True)
                        w_off = batch[1].to(device, non_blocking=True)
                        b_idx = batch[2].to(device, non_blocking=True)
                        b_off = batch[3].to(device, non_blocking=True)
                        stm = batch[4].to(device, non_blocking=True)
                        target = batch[5].to(device, non_blocking=True)
                    pred = model(w_idx, w_off, b_idx, b_off, stm)
                    val_loss = wdl_loss(pred, target).item()
                    tracker.track("val", "loss", val_loss)
                    pbar.set_postfix(**tracker.postfix)
                    pbar.update(1)

        scheduler.step()
        epoch_metrics = tracker.submit_epoch()

        # Keep val_loss visible in progress bar for next epoch
        if "val" in epoch_metrics and "loss" in epoch_metrics["val"]:
            tracker.track_epoch("val_loss", f"{epoch_metrics['val']['loss']:.4f}")

        # Print epoch summary when quiet
        if quiet:
            train_loss = epoch_metrics.get("train", {}).get("loss", float("nan"))
            val_loss = epoch_metrics.get("val", {}).get("loss", float("nan"))
            print(
                f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
            )

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


_ARCH_STRINGS = {
    "halfkp": b"Features=HalfKP(Friend)[41024->256x2]->[32->32]->1",
    "halfkav2": b"Features=HalfKAv2(Friend)[7692->256x2]->[32->32]->1",
}

_ARCH_FT_SIZES = {
    "halfkp": HALFKP_SIZE,
    "halfkav2": HALFKAV2_SIZE,
}


def export_nnue(model: NNUE, path: str):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x7AF32F20))
        f.write(struct.pack("<I", 0))
        arch_str = _ARCH_STRINGS[model.arch]
        f.write(struct.pack("<I", len(arch_str)))
        f.write(arch_str)
        f.write(struct.pack("<I", 0x5D69D5B9))

        # Feature transformer
        bias = (
            np.round(model.ft_bias.detach().cpu().numpy() * FT_QUANT_SCALE)
            .clip(-32768, 32767)
            .astype(np.int16)
        )
        f.write(bias.tobytes())
        weight = (
            np.round(model.ft.weight.detach().cpu().numpy().T * FT_QUANT_SCALE)
            .clip(-32768, 32767)
            .astype(np.int16)
        )
        f.write(weight.T.tobytes())

        f.write(struct.pack("<I", 0))

        # Hidden layers
        # Bias scale must match the dot product scale at each layer.
        # After ClippedReLU + divide-by-NET_SCALE, activations are at scale
        # FT_QUANT_SCALE (127). So dot(input, weights) has scale
        # FT_QUANT_SCALE * WEIGHT_QUANT_SCALE = 8128 for all layers.
        for layer, scale in [
            (model.l1, FT_QUANT_SCALE * WEIGHT_QUANT_SCALE),
            (model.l2, FT_QUANT_SCALE * WEIGHT_QUANT_SCALE),
        ]:
            f.write(
                np.round(layer.bias.detach().cpu().numpy() * scale)
                .astype(np.int32)
                .tobytes()
            )
            f.write(
                np.round(layer.weight.detach().cpu().numpy() * WEIGHT_QUANT_SCALE)
                .clip(-128, 127)
                .astype(np.int8)
                .tobytes()
            )

        # Output (same bias scale as hidden layers: FT_QUANT_SCALE * WEIGHT_QUANT_SCALE)
        f.write(
            np.round(
                model.out.bias.detach().cpu().numpy()
                * FT_QUANT_SCALE
                * WEIGHT_QUANT_SCALE
            )
            .astype(np.int32)
            .tobytes()
        )
        f.write(
            np.round(
                model.out.weight.detach().cpu().numpy().flatten() * WEIGHT_QUANT_SCALE
            )
            .clip(-128, 127)
            .astype(np.int8)
            .tobytes()
        )

    print(f"Exported: {path}")


def load_nnue(path: str) -> dict:
    """Load a quantized .nnue file into numpy arrays.

    Returns a dict with keys: arch, ft_bias, ft_weights, l1_bias, l1_weights,
    l2_bias, l2_weights, out_bias, out_weights.
    """
    with open(path, "rb") as f:
        # Header
        f.read(4)  # version
        f.read(4)  # hash
        arch_len = struct.unpack("<I", f.read(4))[0]
        arch_bytes = f.read(arch_len)
        f.read(4)  # FT header

        # Detect architecture from header string
        arch = "halfkp"  # default
        for name, expected in _ARCH_STRINGS.items():
            if arch_bytes == expected:
                arch = name
                break
        ft_size = _ARCH_FT_SIZES[arch]

        # Feature transformer
        ft_bias = np.frombuffer(f.read(FT_OUT * 2), dtype=np.int16).copy()
        ft_weights = (
            np.frombuffer(f.read(ft_size * FT_OUT * 2), dtype=np.int16)
            .reshape(ft_size, FT_OUT)
            .copy()
        )

        # Network header
        f.read(4)

        # L1: (L1_OUT, L1_IN) = (32, 512)
        l1_bias = np.frombuffer(f.read(L1_OUT * 4), dtype=np.int32).copy()
        l1_weights = (
            np.frombuffer(f.read(L1_OUT * FT_OUT * 2 * 1), dtype=np.int8)
            .reshape(L1_OUT, FT_OUT * 2)
            .copy()
        )

        # L2: (L2_OUT, L2_IN) = (32, 32)
        l2_bias = np.frombuffer(f.read(L2_OUT * 4), dtype=np.int32).copy()
        l2_weights = (
            np.frombuffer(f.read(L2_OUT * L1_OUT * 1), dtype=np.int8)
            .reshape(L2_OUT, L1_OUT)
            .copy()
        )

        # Output: (1, 32)
        out_bias = struct.unpack("<i", f.read(4))[0]
        out_weights = np.frombuffer(f.read(L2_OUT * 1), dtype=np.int8).copy()

    return {
        "arch": arch,
        "ft_bias": ft_bias,
        "ft_weights": ft_weights,
        "l1_bias": l1_bias,
        "l1_weights": l1_weights,
        "l2_bias": l2_bias,
        "l2_weights": l2_weights,
        "out_bias": out_bias,
        "out_weights": out_weights,
    }


def _trunc_div(a: int, b: int) -> int:
    """C-style integer division (truncate toward zero)."""
    return int(a / b) if (a ^ b) >= 0 else -int((-a) / b)


def evaluate_fen_quantized(fen: str, weights: dict) -> int:
    """Evaluate a FEN using quantized integer arithmetic matching C++ NNUE.hpp.

    This replicates the exact computation in Evaluator::evaluate() + forward()
    so we can verify the C++ implementation is correct.

    Args:
        fen: FEN string
        weights: dict from load_nnue()

    Returns:
        Score in centipawns from side-to-move perspective (int).
    """
    NET_SCALE = 64
    FT_SCALE_VAL = 127
    SIGMOID_SCALE_VAL = 400

    # Extract features using the architecture from the weights
    compressed = dummy_chess.compress_fen(fen)
    arch = weights.get("arch", "halfkp")
    if arch == "halfkav2":
        w_idx, w_off, b_idx, b_off, stm = dummy_chess.get_halfkav2_features_batch(
            [compressed]
        )
    else:
        w_idx, w_off, b_idx, b_off, stm = dummy_chess.get_halfkp_features_batch(
            [compressed]
        )

    # Build accumulators (int16, matching C++ refresh_accumulator)
    ft_bias = weights["ft_bias"]
    ft_w = weights["ft_weights"]

    # White perspective accumulator
    w_acc = ft_bias.astype(np.int32).copy()
    for idx in w_idx:
        w_acc += ft_w[int(idx)].astype(np.int32)
    # Clip to int16 range (accumulator is int16 in C++)
    w_acc = np.clip(w_acc, -32768, 32767).astype(np.int16)

    # Black perspective accumulator
    b_acc = ft_bias.astype(np.int32).copy()
    for idx in b_idx:
        b_acc += ft_w[int(idx)].astype(np.int32)
    b_acc = np.clip(b_acc, -32768, 32767).astype(np.int16)

    # ClippedReLU: int16 -> int8, clamp [0, 127]
    # Concatenate STM first, NSTM second (matching C++ forward)
    stm_val = int(stm[0])
    if stm_val == 0:  # white to move
        stm_acc, nstm_acc = w_acc, b_acc
    else:  # black to move
        stm_acc, nstm_acc = b_acc, w_acc

    ft_out = np.concatenate(
        [
            np.clip(stm_acc, 0, 127).astype(np.int8),
            np.clip(nstm_acc, 0, 127).astype(np.int8),
        ]
    )

    # L1: 512 -> 32
    l1_out = np.empty(L1_OUT, dtype=np.int8)
    for j in range(L1_OUT):
        s = int(weights["l1_bias"][j])
        s += int(
            np.dot(ft_out.astype(np.int32), weights["l1_weights"][j].astype(np.int32))
        )
        # C-style: clamp(sum / 64, 0, 127)
        s = max(0, min(127, _trunc_div(s, NET_SCALE)))
        l1_out[j] = s

    # L2: 32 -> 32
    l2_out = np.empty(L2_OUT, dtype=np.int8)
    for j in range(L2_OUT):
        s = int(weights["l2_bias"][j])
        s += int(
            np.dot(l1_out.astype(np.int32), weights["l2_weights"][j].astype(np.int32))
        )
        s = max(0, min(127, _trunc_div(s, NET_SCALE)))
        l2_out[j] = s

    # Output: 32 -> 1
    raw = int(weights["out_bias"])
    raw += int(np.dot(l2_out.astype(np.int32), weights["out_weights"].astype(np.int32)))

    # to_centipawns: raw * 400 / 8128 (C-style truncation)
    cp = _trunc_div(raw * SIGMOID_SCALE_VAL, FT_SCALE_VAL * NET_SCALE)

    # Negate for black STM (network outputs white's perspective)
    if stm_val == 1:
        cp = -cp

    return cp


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
        # Peek at state dict to detect arch from ft.weight shape
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        ft_size = state["ft.weight"].shape[0]
        arch = "halfkav2" if ft_size == HALFKAV2_SIZE else "halfkp"
        model = NNUE(arch=arch)
        model.load_state_dict(state)
        model.eval()

    device = _get_model_device(model)

    # Use correct feature extractor for the model's architecture
    fen_str = fen if isinstance(fen, str) else dummy_chess.decompress_fen(fen)
    if model.arch == "halfkav2":
        w_feats, b_feats, stm = dummy_chess.get_halfkav2_features(fen_str)
    else:
        w_feats, b_feats, stm = get_halfkp_features(fen_str)

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
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        ft_size = state["ft.weight"].shape[0]
        arch = "halfkav2" if ft_size == HALFKAV2_SIZE else "halfkp"
        model = NNUE(arch=arch)
        model.load_state_dict(state)
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

    # Extract features for all positions using correct architecture
    if model.arch == "halfkav2":
        batch = [dummy_chess.get_halfkav2_features(f) for f in fen_strs]
    else:
        batch = [get_halfkp_features(f) for f in fen_strs]

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
        "--arch",
        choices=["halfkp", "halfkav2"],
        default="halfkp",
        help="NNUE architecture: halfkp (41K features) or halfkav2 (7.7K features, king buckets)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bar, only print epoch summaries",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=4,
        help="Number of batches to prefetch (0 to disable)",
    )
    parser.add_argument(
        "--grad-acc-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch = batch_size * grad_acc_steps.",
    )
    parser.add_argument(
        "--gpu-extract",
        action="store_true",
        help="Extract features on GPU using CUDA kernels (requires _extract_cuda extension and CUDA device)",
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

    # Build datasets — GPU extraction or CPU extraction
    if args.gpu_extract:
        train_dataset = GpuExtractDataset(
            args.data,
            batch_size=args.batch_size,
            split="train",
            split_config=split_config,
            arch=args.arch,
            flip_augment=args.flip_augment,
        )
        val_dataset = GpuExtractDataset(
            args.data,
            batch_size=args.batch_size,
            split="val",
            split_config=split_config,
            arch=args.arch,
            flip_augment=False,
        )
    else:
        train_dataset = ParquetDataset(
            args.data,
            batch_size=args.batch_size,
            split="train",
            split_config=split_config,
            arch=args.arch,
            flip_augment=args.flip_augment,
            prefetch=args.prefetch,
        )
        val_dataset = ParquetDataset(
            args.data,
            batch_size=args.batch_size,
            split="val",
            split_config=split_config,
            arch=args.arch,
            flip_augment=False,
            prefetch=args.prefetch,
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
        quiet=args.quiet,
        accum_steps=args.grad_acc_steps,
    )
