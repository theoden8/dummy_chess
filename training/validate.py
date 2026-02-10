#!/usr/bin/env python3
"""
Validate parquet file integrity by reading row-group by row-group.

Uses O(row_group_size) memory instead of O(file_size).

Usage:
    uv run python validate.py data/evals.parquet
    uv run python validate.py data/*.parquet
"""

import argparse
import sys

import pyarrow.parquet
import tqdm.auto


def validate_parquet(path: str, verbose: bool = False, quiet: bool = False) -> bool:
    """
    Validate parquet file by reading each row group.

    Returns True if valid, False if corrupt.
    """
    try:
        pf = pyarrow.parquet.ParquetFile(path)
    except Exception as e:
        tqdm.auto.tqdm.write(f"CORRUPT (metadata): {path}")
        tqdm.auto.tqdm.write(f"  {e}")
        return False

    num_row_groups = pf.metadata.num_row_groups

    pbar = tqdm.auto.tqdm(
        range(num_row_groups),
        desc=path,
        unit=" rg",
        disable=quiet,
        leave=verbose,
    )

    for i in pbar:
        try:
            table = pf.read_row_group(i)
            del table
        except Exception as e:
            pbar.close()
            tqdm.auto.tqdm.write(f"CORRUPT (row group {i}): {path}")
            tqdm.auto.tqdm.write(f"  {e}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Validate parquet file integrity")
    parser.add_argument("files", nargs="+", help="Parquet files to validate")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Keep progress bars"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Only show errors")
    args = parser.parse_args()

    valid = 0
    corrupt = 0

    for path in tqdm.auto.tqdm(
        args.files, desc="Files", unit=" files", disable=len(args.files) == 1
    ):
        if validate_parquet(path, verbose=args.verbose, quiet=args.quiet):
            valid += 1
        else:
            corrupt += 1

    if len(args.files) > 1 and not args.quiet:
        tqdm.auto.tqdm.write(f"\nTotal: {valid} valid, {corrupt} corrupt")

    sys.exit(1 if corrupt > 0 else 0)


if __name__ == "__main__":
    main()
