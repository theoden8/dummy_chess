#!/usr/bin/env python3
"""Test script for pl_scan_ndjson_zstd with 2GB memory limit."""

import argparse
import resource
import time

import polars as pl
from preprocess_data import pl_scan_ndjson_zstd

GB = 1024 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (2 * GB, 2 * GB))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--sample", type=int, default=10, help="Sample N rows with skip-based sampling"
)
parser.add_argument(
    "--skip", type=int, default=10000, help="Skip interval for fast sampling"
)
parser.add_argument(
    "--reservoir", action="store_true", help="Use reservoir sampling (slow, full scan)"
)
parser.add_argument(
    "--length", action="store_true", help="Count rows (slow, full scan)"
)
args = parser.parse_args()

path = "./data/lichess_db_eval.jsonl.zst"
print(f"Scanning: {path}")

lf = pl_scan_ndjson_zstd(path)
print(f"Columns: {lf.collect_schema().names()}")


def format_row(row):
    """Format a row for display."""
    evals = row["evals"]
    best = evals[0]["pvs"][0] if evals and evals[0]["pvs"] else {}
    cp = best.get("cp")
    mate = best.get("mate")
    score_str = (
        f"cp={cp}" if cp is not None else f"mate={mate}" if mate is not None else "?"
    )
    return f"{row['fen'][:50]}... | {score_str}"


# Test head (fast)
print("\nHead (5 rows):")
df = lf.head(5)
for i in range(len(df)):
    row = df.row(i, named=True)
    print(f"  [{i}] {format_row(row)}")

# Test sampling
if args.sample > 0:
    if args.reservoir:
        print(f"\nSampling {args.sample} rows (reservoir - full scan, slow!)...")
        t0 = time.perf_counter()
        sample_df = lf.sample(args.sample, seed=42)
    else:
        print(f"\nSampling {args.sample} rows (skip={args.skip}, fast)...")
        t0 = time.perf_counter()
        sample_df = lf.sample(args.sample, skip=args.skip)
    t1 = time.perf_counter()
    print(f"Sample time: {t1 - t0:.2f}s")
    for i in range(len(sample_df)):
        row = sample_df.row(i, named=True)
        print(f"  [{i}] {format_row(row)}")

# Test length (full scan - slowest operation)
if args.length:
    print("\nCounting rows (full scan)...")
    t0 = time.perf_counter()
    length = len(lf)
    t1 = time.perf_counter()
    print(f"Length: {length} (took {t1 - t0:.2f}s)")
