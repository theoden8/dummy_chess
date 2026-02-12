#!/usr/bin/env python3
"""Diagnose data loading bottleneck."""

import time
import pyarrow.parquet
import numpy as np
import dummy_chess

print("=== Diagnosing data loading speed ===\n")

# Test 1: Raw pyarrow reading
print("Test 1: PyArrow row group reading")
pf = pyarrow.parquet.ParquetFile("data/preprocessed_evaluations_shuffled.parquet")
t0 = time.perf_counter()
for i in range(10):
    table = pf.read_row_group(i, columns=["fen", "score"])
elapsed = time.perf_counter() - t0
print(f"  {10 / elapsed:.1f} row_groups/sec, {table.num_rows:,} rows each")

# Test 2: FEN extraction to Python list
print("\nTest 2: FEN to_pylist()")
table = pf.read_row_group(0, columns=["fen", "score"])
t0 = time.perf_counter()
for _ in range(10):
    fens = table.column("fen").to_pylist()
elapsed = time.perf_counter() - t0
print(f"  {10 / elapsed:.1f} calls/sec, {len(fens):,} fens each")

# Test 3: C++ feature extraction
print("\nTest 3: C++ get_halfkp_features_batch()")
fens_batch = fens[:8192]
t0 = time.perf_counter()
for _ in range(10):
    w_idx, w_off, b_idx, b_off, stm = dummy_chess.get_halfkp_features_batch(
        fens_batch, False
    )
elapsed = time.perf_counter() - t0
print(f"  {10 / elapsed:.1f} calls/sec, 8192 fens each")

# Test 4: Feature extraction with flip (2x work)
print("\nTest 4: C++ feature extraction with flip (2x)")
t0 = time.perf_counter()
for _ in range(10):
    w_idx, w_off, b_idx, b_off, stm = dummy_chess.get_halfkp_features_batch(
        fens_batch, False
    )
    w_idx_f, w_off_f, b_idx_f, b_off_f, stm_f = dummy_chess.get_halfkp_features_batch(
        fens_batch, True
    )
elapsed = time.perf_counter() - t0
print(f"  {10 / elapsed:.1f} calls/sec (both normal + flipped)")

# Test 5: Full batch pipeline
print("\nTest 5: Full batch pipeline (read + extract + tensorize)")
import torch

t0 = time.perf_counter()
for i in range(10):
    table = pf.read_row_group(i % pf.metadata.num_row_groups, columns=["fen", "score"])
    fens = table.column("fen").to_pylist()[:8192]
    scores = table.column("score").to_numpy()[:8192].astype(np.float32)
    w_idx, w_off, b_idx, b_off, stm = dummy_chess.get_halfkp_features_batch(fens, False)
    batch = (
        torch.from_numpy(w_idx.astype(np.int64)),
        torch.from_numpy(w_off.astype(np.int64)),
        torch.from_numpy(b_idx.astype(np.int64)),
        torch.from_numpy(b_off.astype(np.int64)),
        torch.from_numpy(stm.astype(np.int64)),
        torch.from_numpy(scores).unsqueeze(1),
    )
elapsed = time.perf_counter() - t0
print(f"  {10 / elapsed:.1f} batches/sec")

print("\n=== Done ===")
print("\nExpected speeds (on fast system):")
print("  PyArrow read: ~50+ row_groups/sec")
print("  FEN to_pylist: ~20+ calls/sec")
print("  Feature extraction: ~100+ calls/sec")
print("  Full pipeline: ~30+ batches/sec")
