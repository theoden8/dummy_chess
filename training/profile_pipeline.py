#!/usr/bin/env python3
"""Profile each stage of the training pipeline to find bottlenecks."""

import time
import torch
import pyarrow.parquet
import dummy_chess
import numpy

# Config
PARQUET_FILE = "data/preprocessed_evaluations.parquet"
BATCH_SIZE = 8192
N_ITERATIONS = 10

print(f"Profiling with batch_size={BATCH_SIZE}, iterations={N_ITERATIONS}")
print("=" * 60)

# Load one batch worth of data
pf = pyarrow.parquet.ParquetFile(PARQUET_FILE)
table = pf.read_row_group(0, columns=["fen", "score"])
fens = table.column("fen").to_pylist()[:BATCH_SIZE]
scores = table.column("score").to_numpy()[:BATCH_SIZE].astype(numpy.float32)

print(f"Loaded {len(fens)} samples from {PARQUET_FILE}")
print("=" * 60)

# 1. PyArrow read
t0 = time.perf_counter()
for _ in range(N_ITERATIONS):
    table = pf.read_row_group(0, columns=["fen", "score"])
    fens = table.column("fen").to_pylist()[:BATCH_SIZE]
    scores = table.column("score").to_numpy()[:BATCH_SIZE]
t1 = time.perf_counter()
print(
    f"1. PyArrow read + to_pylist:  {(t1 - t0) / N_ITERATIONS * 1000:6.1f}ms per batch"
)

# 2. C++ feature extraction
t0 = time.perf_counter()
for _ in range(N_ITERATIONS):
    w_idx, w_off, b_idx, b_off, stm = dummy_chess.get_halfkp_features_batch(fens, False)
t1 = time.perf_counter()
cpp_time = (t1 - t0) / N_ITERATIONS * 1000
print(f"2. C++ feature extraction:    {cpp_time:6.1f}ms per batch")

# 3. Tensor conversion (numpy -> torch)
t0 = time.perf_counter()
for _ in range(N_ITERATIONS):
    w_idx_t = torch.from_numpy(w_idx.astype(numpy.int64))
    w_off_t = torch.from_numpy(w_off.astype(numpy.int64))
    b_idx_t = torch.from_numpy(b_idx.astype(numpy.int64))
    b_off_t = torch.from_numpy(b_off.astype(numpy.int64))
    stm_t = torch.from_numpy(stm.astype(numpy.int64))
    scores_t = torch.from_numpy(scores.astype(numpy.float32))
t1 = time.perf_counter()
print(
    f"3. Tensor conversion:         {(t1 - t0) / N_ITERATIONS * 1000:6.1f}ms per batch"
)

# 4. GPU transfer
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Warm up
    w_idx_t.to(device)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_ITERATIONS):
        w_idx_t.to(device)
        w_off_t.to(device)
        b_idx_t.to(device)
        b_off_t.to(device)
        stm_t.to(device)
        scores_t.to(device)
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(
        f"4. GPU transfer:              {(t1 - t0) / N_ITERATIONS * 1000:6.1f}ms per batch"
    )
else:
    print(f"4. GPU transfer:              N/A (no CUDA)")

print("=" * 60)

# Summary
batches_per_sec = 1000 / cpp_time
print(f"\nC++ extraction rate: {batches_per_sec:.1f} batches/sec")
print(f"At 30 batches/sec target, C++ should take: {1000 / 30:.1f}ms")
print(f"Current C++ time: {cpp_time:.1f}ms")

if cpp_time > 50:
    print("\n⚠️  C++ feature extraction is likely the bottleneck!")
    print("   Check if dummy_chess was built in Release mode:")
    print("   uv sync --reinstall-package dummy-chess")
