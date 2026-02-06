#!/usr/bin/env python3
"""
Preprocess Lichess data for NNUE training.

Supports two data sources:
1. Puzzles: lichess_db_puzzle.csv.zst - uses Stockfish or material eval
2. Evaluations: lichess_db_eval.jsonl.zst - uses pre-computed Stockfish evals

Generates train/val/test splits with fen, score, and knodes fields.
"""

import argparse
import io
import random
import sys
from pathlib import Path
from typing import Iterator

import chess
import chess.engine
import orjson
import polars as pl
import zstandard as zstd
from tqdm.auto import tqdm


class LazyNdjsonZstd:
    """
    LazyFrame-like interface for zstd-compressed NDJSON files.

    Uses streaming decompression to stay within memory limits.
    Provides a subset of LazyFrame methods for compatibility.
    """

    def __init__(self, path: str | Path, batch_size: int = 100000):
        self._path = Path(path)
        self._batch_size = batch_size
        self._schema: pl.Schema | None = None

    def _open_stream(self) -> tuple:
        """Open zstd decompression stream for reading."""
        f = open(self._path, "rb")
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        text = io.TextIOWrapper(reader, encoding="utf-8")
        return (f, reader, text)

    def _close_stream(self, stream_info: tuple) -> None:
        """Close the stream opened by _open_stream."""
        f, reader, text = stream_info
        text.close()
        reader.close()
        f.close()

    def _iter_lines(self) -> Iterator[dict]:
        """Iterate over parsed JSON lines."""
        stream_info = self._open_stream()
        try:
            for line in stream_info[2]:
                yield orjson.loads(line)
        finally:
            self._close_stream(stream_info)

    def collect_schema(self) -> pl.Schema:
        """Get schema from first row."""
        if self._schema is None:
            first = next(self._iter_lines())
            df = pl.from_dicts([first])
            self._schema = df.schema
        return pl.Schema(self._schema)

    def head(self, n: int = 10) -> pl.DataFrame:
        """Collect first n rows."""
        import itertools

        rows = list(itertools.islice(self._iter_lines(), n))
        return pl.from_dicts(rows)

    def select(self, *exprs) -> "LazyNdjsonZstd":
        """Select columns (returns self for chaining, applied at collect)."""
        return self

    def collect(self) -> pl.DataFrame:
        """Collect all data into a DataFrame (warning: may use lots of memory)."""
        return pl.from_dicts(list(self._iter_lines()))

    def iter_batches(self, batch_size: int | None = None) -> Iterator[pl.DataFrame]:
        """Iterate over batches of data as DataFrames."""
        batch_size = batch_size or self._batch_size
        batch = []
        for row in self._iter_lines():
            batch.append(row)
            if len(batch) >= batch_size:
                yield pl.from_dicts(batch)
                batch = []
        if batch:
            yield pl.from_dicts(batch)

    def iter_rows(self) -> Iterator[dict]:
        """Iterate over rows as dicts (streaming, memory-efficient)."""
        yield from self._iter_lines()

    def sample(
        self,
        n: int,
        seed: int | None = None,
        skip: int | None = None,
        max_rows: int | None = None,
    ) -> pl.DataFrame:
        """
        Sample n rows from the file.

        Two modes:
        1. skip=None (default): Reservoir sampling (uniform random, requires full scan)
        2. skip=N: Take every Nth row (fast, biased toward file start)

        Args:
            n: Number of rows to sample
            seed: Random seed for reservoir sampling
            skip: If set, take every `skip` rows (fast mode, ignores seed)
            max_rows: Stop after scanning this many rows (for skip mode)

        Returns:
            DataFrame with sampled rows
        """
        if skip is not None:
            # Fast skip-based sampling
            rows = []
            stream_info = self._open_stream()
            try:
                for i, line in enumerate(stream_info[2]):
                    if max_rows and i >= max_rows:
                        break
                    if i % skip == 0:
                        rows.append(orjson.loads(line))
                        if len(rows) >= n:
                            break
            finally:
                self._close_stream(stream_info)
            return pl.from_dicts(rows)

        # Reservoir sampling (Algorithm R) - single pass, uniform random
        rng = random.Random(seed)
        reservoir: list[dict] = []
        stream_info = self._open_stream()
        try:
            for i, line in enumerate(stream_info[2]):
                row = orjson.loads(line)
                if i < n:
                    reservoir.append(row)
                else:
                    j = rng.randint(0, i)
                    if j < n:
                        reservoir[j] = row
        finally:
            self._close_stream(stream_info)

        return pl.from_dicts(reservoir)

    def __len__(self) -> int:
        """Count total rows (requires full scan)."""
        f, reader, text = self._open_stream()
        try:
            count = sum(1 for _ in text)
        finally:
            text.close()
            reader.close()
            f.close()
        return count


def pl_scan_ndjson_zstd(path: str | Path, batch_size: int = 100000) -> LazyNdjsonZstd:
    """
    Scan a zstd-compressed NDJSON file.

    Returns a LazyNdjsonZstd object that provides LazyFrame-like interface
    while streaming data to stay within memory limits.
    """
    return LazyNdjsonZstd(path, batch_size)


def pl_scan_csv_zstd(path: str | Path) -> pl.LazyFrame:
    """Scan a zstd-compressed CSV file as a polars LazyFrame."""
    f = open(path, "rb")
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(f)
    return pl.scan_csv(reader)


# =============================================================================
# Puzzle processing
# =============================================================================

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def material_eval(board: chess.Board) -> int:
    """Simple material count in centipawns."""
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = PIECE_VALUES.get(piece.piece_type, 0)
            score += val if piece.color == chess.WHITE else -val
    return score


def stockfish_eval(board: chess.Board, engine, depth: int = 12) -> tuple[int, int]:
    """
    Stockfish evaluation.

    Returns:
        (score_cp, nodes) - score in centipawns from white's perspective, node count
    """
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white()
    nodes = info.get("nodes", 0)

    if score.is_mate():
        cp = 10000 if score.mate() > 0 else -10000
    else:
        cp = score.score()

    return cp, nodes


def process_puzzle_row(
    row: dict, engine=None, depth: int = 12
) -> tuple[str, int, int] | None:
    """
    Process single puzzle row.

    Returns:
        (fen, score, knodes) or None if invalid
    """
    try:
        fen = row.get("FEN")
        moves_str = row.get("Moves")
        if not fen or not moves_str:
            return None

        moves = moves_str.split()
        board = chess.Board(fen)
        for m in moves:
            move = chess.Move.from_uci(m)
            if move not in board.legal_moves:
                return None
            board.push(move)

        if engine:
            score, nodes = stockfish_eval(board, engine, depth)
            knodes = nodes // 1000
        else:
            score = material_eval(board)
            knodes = 0

        score = max(-15000, min(15000, score))
        return board.fen(), score, knodes
    except Exception:
        return None


# =============================================================================
# Evaluation data processing
# =============================================================================


def process_eval_row(row: dict) -> tuple[str, int, int] | None:
    """
    Process single evaluation row from lichess_db_eval.jsonl.zst.

    The eval data has format:
    {
        "fen": "...",
        "evals": [
            {
                "pvs": [{"cp": 69, "line": "..."}, ...] or [{"mate": 5, "line": "..."}, ...],
                "knodes": 4189972,
                "depth": 46
            },
            ...
        ]
    }

    We take the best eval (first PV of first eval entry, which has highest depth).

    Returns:
        (fen, score, knodes) or None if invalid
    """
    try:
        fen = row.get("fen")
        evals = row.get("evals")

        if not fen or not evals:
            return None

        # Get first eval entry (highest depth)
        eval_entry = evals[0]
        pvs = eval_entry.get("pvs", [])
        knodes = eval_entry.get("knodes", 0)

        if not pvs:
            return None

        # Get best move evaluation
        best_pv = pvs[0]
        cp = best_pv.get("cp")
        mate = best_pv.get("mate")

        if cp is not None:
            score = cp
        elif mate is not None:
            # Convert mate to large cp value
            # Positive mate = winning, negative = losing
            score = 10000 if mate > 0 else -10000
        else:
            return None

        # Clamp score
        score = max(-15000, min(15000, score))

        # Convert knodes to int (it's already in thousands in the data)
        knodes = knodes // 1000 if knodes > 1000 else knodes

        return fen, score, knodes
    except Exception:
        return None


# =============================================================================
# Main processing functions
# =============================================================================


def process_puzzles(
    input_path: Path,
    max_rows: int | None,
    stockfish_path: str | None,
    depth: int,
) -> list[tuple[str, int, int]]:
    """Process puzzle data with optional Stockfish evaluation."""
    # Setup stockfish
    engine = None
    if stockfish_path:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            print(f"Using Stockfish: {stockfish_path} (depth={depth})")
        except Exception as e:
            print(f"Stockfish failed ({e}), using material eval")

    # Load puzzles
    print(f"Loading {input_path}...")
    if str(input_path).endswith(".zst"):
        lf = pl_scan_csv_zstd(input_path)
    else:
        lf = pl.scan_csv(input_path)

    if max_rows:
        lf = lf.head(max_rows)

    # Process with progress bar
    print("Processing puzzles...")
    data = []
    rows = list(lf.collect().iter_rows(named=True))
    for row in tqdm(rows, desc="Puzzles"):
        result = process_puzzle_row(row, engine, depth)
        if result:
            data.append(result)

    if engine:
        engine.quit()

    return data


def process_evals(
    input_path: Path,
    max_rows: int | None,
) -> list[tuple[str, int, int]]:
    """Process evaluation data in streaming manner."""
    print(f"Loading {input_path}...")
    lf = pl_scan_ndjson_zstd(input_path)

    # Process with streaming and progress bar
    print("Processing evaluations...")
    data = []
    count = 0

    pbar = tqdm(desc="Evals", unit=" rows")
    for row in lf.iter_rows():
        if max_rows and count >= max_rows:
            break

        result = process_eval_row(row)
        if result:
            data.append(result)

        count += 1
        pbar.update(1)

    pbar.close()
    return data


def save_splits(
    data: list[tuple[str, int, int]],
    output_dir: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    """Shuffle data and save train/val/test splits."""
    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    splits = {
        "train": data[:n_train],
        "val": data[n_train : n_train + n_val],
        "test": data[n_train + n_val :],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, rows in splits.items():
        out_df = pl.DataFrame(rows, schema=["fen", "score", "knodes"], orient="row")
        path = output_dir / f"{name}.csv.gz"
        out_df.write_csv(path, compression="gzip")
        print(f"{name}: {len(out_df)} -> {path}")

    # Stats
    all_scores = [s for _, s, _ in data]
    all_knodes = [k for _, _, k in data]
    print(
        f"\nScore stats: min={min(all_scores)}, max={max(all_scores)}, "
        f"mean={sum(all_scores) / len(all_scores):.0f}"
    )
    print(
        f"Knodes stats: min={min(all_knodes)}, max={max(all_knodes)}, "
        f"mean={sum(all_knodes) / len(all_knodes):.0f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Lichess data for NNUE training"
    )
    parser.add_argument(
        "source",
        choices=["puzzles", "evals"],
        help="Data source: 'puzzles' or 'evals'",
    )
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help="Input file (default: data/lichess_db_puzzle.csv.zst or data/lichess_db_eval.jsonl.zst)",
    )
    parser.add_argument("-o", "--output-dir", default="data")
    parser.add_argument(
        "-s", "--stockfish", default=None, help="Path to stockfish (puzzles only)"
    )
    parser.add_argument(
        "-d", "--depth", type=int, default=12, help="Stockfish depth (puzzles only)"
    )
    parser.add_argument(
        "-n", "--max", type=int, default=None, help="Max rows to process"
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--toy", action="store_true", help="Generate toy dataset (10k rows)"
    )
    args = parser.parse_args()

    # Set defaults based on source
    if args.input is None:
        if args.source == "puzzles":
            args.input = "data/lichess_db_puzzle.csv.zst"
        else:
            args.input = "data/lichess_db_eval.jsonl.zst"

    if args.toy:
        args.max = 35000 if args.source == "puzzles" else 10000
        args.output_dir = f"data/toy_{args.source}"

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {args.input} not found")
        if args.source == "puzzles":
            print(
                "Download: wget https://database.lichess.org/lichess_db_puzzle.csv.zst"
            )
        else:
            print(
                "Download: wget https://database.lichess.org/lichess_db_eval.jsonl.zst"
            )
        sys.exit(1)

    # Process based on source
    if args.source == "puzzles":
        data = process_puzzles(input_path, args.max, args.stockfish, args.depth)
    else:
        data = process_evals(input_path, args.max)

    print(f"Valid: {len(data)}")

    # Save splits
    save_splits(
        data,
        Path(args.output_dir),
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )


if __name__ == "__main__":
    main()
