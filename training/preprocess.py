#!/usr/bin/env python3
"""
Preprocess data for NNUE training.

Supports three data sources:
1. puzzles: lichess_db_puzzle.csv.zst - uses UCI engine or material eval
2. evals: lichess_db_eval.jsonl.zst - uses pre-computed engine evals from lichess
3. endgames: generates random endgame positions using Syzygy tablebases

Outputs a single CSV file with columns: fen, score

STYLE: Do NOT use `from X import Y` - use fully qualified names (e.g. pathlib.Path, not Path)
STYLE: Do NOT use import aliases (e.g. `import numpy as np`) - use full module names
"""

import argparse
import dataclasses
import hashlib
import io
import itertools
import pathlib
import random
import shutil
import sys
import tempfile
import typing

import chess
import chess.engine
import chess.syzygy
import dummy_chess
import orjson
import polars
import pyarrow.parquet
import tqdm.auto
import zstandard


# =============================================================================
# LazyNdjsonZstd - streaming zstd NDJSON reader
# =============================================================================


class LazyNdjsonZstd:
    """
    LazyFrame-like interface for zstd-compressed NDJSON files.

    Uses streaming decompression to stay within memory limits.
    """

    def __init__(self, path: str, batch_size: int = 100000):
        self._path = pathlib.Path(path)
        self._batch_size = batch_size
        self._schema: polars.Schema | None = None

    def _open_stream(self) -> tuple:
        """Open zstd decompression stream for reading."""
        f = open(self._path, "rb")
        dctx = zstandard.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        text = io.TextIOWrapper(reader, encoding="utf-8")
        return (f, reader, text)

    def _close_stream(self, stream_info: tuple) -> None:
        """Close the stream opened by _open_stream."""
        f, reader, text = stream_info
        text.close()
        reader.close()
        f.close()

    def _iter_lines(self) -> typing.Iterator[dict]:
        """Iterate over parsed JSON lines."""
        stream_info = self._open_stream()
        try:
            for line in stream_info[2]:
                yield orjson.loads(line)
        finally:
            self._close_stream(stream_info)

    def collect_schema(self) -> polars.Schema:
        """Get schema from first row."""
        if self._schema is None:
            first = next(self._iter_lines())
            df = polars.from_dicts([first])
            self._schema = df.schema
        return polars.Schema(self._schema)

    def head(self, n: int = 10) -> polars.DataFrame:
        """Collect first n rows."""
        rows = list(itertools.islice(self._iter_lines(), n))
        return polars.from_dicts(rows)

    def iter_rows(self) -> typing.Iterator[dict]:
        """Iterate over rows as dicts (streaming, memory-efficient)."""
        yield from self._iter_lines()

    def sample(
        self,
        n: int,
        seed: int | None = None,
        skip: int | None = None,
        max_rows: int | None = None,
    ) -> polars.DataFrame:
        """
        Sample n rows from the file.

        Two modes:
        1. skip=None (default): Reservoir sampling (uniform random, requires full scan)
        2. skip=N: Take every Nth row (fast, biased toward file start)
        """
        if skip is not None:
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
            return polars.from_dicts(rows)

        # Reservoir sampling
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
        return polars.from_dicts(reservoir)

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


def pl_scan_ndjson_zstd(path: str, batch_size: int = 100000) -> LazyNdjsonZstd:
    """Scan a zstd-compressed NDJSON file."""
    return LazyNdjsonZstd(path, batch_size)


def pl_scan_csv_zstd(path: str) -> polars.LazyFrame:
    """Scan a zstd-compressed CSV file as a polars LazyFrame."""
    f = open(path, "rb")
    dctx = zstandard.ZstdDecompressor()
    reader = dctx.stream_reader(f)
    return polars.scan_csv(reader)


# =============================================================================
# Puzzle processing
# =============================================================================


def uci_eval(
    board: chess.Board,
    engine,
    depth: int = 12,
) -> tuple[int, int, int]:
    """UCI engine evaluation. Returns (score_cp, depth, knodes)."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white()
    if score.is_mate():
        cp = 10000 if score.mate() > 0 else -10000
    else:
        cp = score.score() or 0
    actual_depth = info.get("depth", depth)
    nodes = info.get("nodes", 0)
    knodes = nodes // 1000  # Convert to knodes to match lichess format
    return cp, actual_depth, knodes


def process_puzzle_row(
    row: dict, engine, depth: int = 12
) -> list[tuple[str, int, int, int]]:
    """
    Process single puzzle row. Returns list of (fen_str, score, depth, knodes).

    Extracts two positions:
    1. Starting position (FEN) - before any moves
    2. After move 1 - the opponent's "mistake" that creates the puzzle
    """
    fen = row.get("FEN")
    moves_str = row.get("Moves")
    if not fen or not moves_str:
        return []

    moves = moves_str.split()
    if not moves:
        return []

    results: list[tuple[str, int, int, int]] = []

    # Position 1: Starting position
    board = chess.Board(fen)
    score, out_depth, out_knodes = uci_eval(board, engine, depth)
    score = max(-15000, min(15000, score))
    results.append((board.fen(), score, out_depth, out_knodes))

    # Position 2: After move 1 (opponent's mistake)
    move = chess.Move.from_uci(moves[0])
    if move not in board.legal_moves:
        return results  # Return just the starting position
    board.push(move)

    score, out_depth, out_knodes = uci_eval(board, engine, depth)
    score = max(-15000, min(15000, score))
    results.append((board.fen(), score, out_depth, out_knodes))

    return results


def process_puzzles(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    max_rows: int | None,
    engine_path: str,
    depth: int,
    tablebase_path: str | None = None,
    batch_size: int = 100000,
) -> int:
    """
    Process puzzle data, streaming to parquet in batches.

    Requires a UCI engine for evaluation.

    Returns the number of rows written.
    """
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    print(f"Using UCI engine: {engine_path} (depth={depth})")

    if tablebase_path:
        engine.configure({"SyzygyPath": tablebase_path})
        print(f"Using tablebases: {tablebase_path}")

    print(f"Loading {input_path}...")
    if str(input_path).endswith(".zst"):
        lf = pl_scan_csv_zstd(input_path)
    else:
        lf = polars.scan_csv(input_path)

    if max_rows:
        lf = lf.head(max_rows)

    # Pre-count total rows for progress bar (streaming to avoid memory issues)
    total_rows = lf.select(polars.len()).collect(engine="streaming").item()
    if max_rows:
        total_rows = min(total_rows, max_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp directory for batch parquet files
    tmp_dir = tempfile.mkdtemp(prefix="puzzles_")

    print("Processing puzzles...")
    count = 0
    written = 0
    batch_num = 0
    batch: list[tuple[str, int, int, int]] = []  # FEN strings, not compressed

    def flush_batch():
        nonlocal batch, batch_num
        if not batch:
            return
        # Batch compress all FENs at once
        fens = [row[0] for row in batch]
        compressed_fens = dummy_chess.compress_fens_batch(fens)
        compressed_batch = [
            (compressed_fens[i], batch[i][1], batch[i][2], batch[i][3])
            for i in range(len(batch))
        ]
        df = polars.DataFrame(
            compressed_batch,
            schema={
                "fen": polars.Binary,
                "score": polars.Int64,
                "depth": polars.Int64,
                "knodes": polars.Int64,
            },
            orient="row",
        )
        df.write_parquet(f"{tmp_dir}/batch_{batch_num:06d}.parquet")
        batch_num += 1
        batch = []

    pbar = tqdm.auto.tqdm(total=total_rows, desc="Puzzles", unit=" rows")
    for df_batch in lf.collect_batches():
        for row in df_batch.iter_rows(named=True):
            if max_rows and count >= max_rows:
                break

            results = process_puzzle_row(row, engine, depth)
            batch.extend(results)
            written += len(results)

            if len(batch) >= batch_size:
                flush_batch()

            count += 1
            pbar.update(1)

        if max_rows and count >= max_rows:
            break

    pbar.close()
    flush_batch()  # Write remaining

    if engine:
        engine.quit()

    print(f"Written {written} rows in {batch_num} batches")

    if batch_num == 0:
        print("No data to write")
        shutil.rmtree(tmp_dir)
        return 0

    # Combine batches and write to output
    print(f"Writing to {output_path}...")
    lf_out = polars.scan_parquet(f"{tmp_dir}/*.parquet")

    out_str = str(output_path)
    if out_str.endswith(".parquet") or not any(
        out_str.endswith(ext) for ext in [".csv", ".csv.gz", ".csv.zst"]
    ):
        lf_out.sink_parquet(output_path)
    else:
        # CSV requires collect
        combined = lf_out.collect(engine="streaming")
        if out_str.endswith(".csv.gz"):
            combined.write_csv(output_path, compression="gzip")
        elif out_str.endswith(".csv.zst"):
            combined.write_csv(output_path, compression="zstd")
        else:
            combined.write_csv(output_path)

    # Cleanup
    shutil.rmtree(tmp_dir)

    print(f"Saved {written} rows -> {output_path}")
    return written


# =============================================================================
# Evaluation data processing
# =============================================================================


def process_eval_row(row: dict) -> tuple[str, int, int, int] | None:
    """Process single evaluation row from lichess_db_eval.jsonl.zst."""
    fen = row.get("fen")
    evals = row.get("evals")

    if not fen or not evals:
        return None

    # Pick eval with highest depth
    eval_entry = max(evals, key=lambda e: e.get("depth", 0))
    depth = eval_entry.get("depth", 0)
    knodes = eval_entry.get("knodes", 0)
    pvs = eval_entry.get("pvs", [])

    if not pvs:
        return None

    best_pv = pvs[0]
    cp = best_pv.get("cp")
    mate = best_pv.get("mate")

    if cp is not None:
        score = cp
    elif mate is not None:
        score = 10000 if mate > 0 else -10000
    else:
        return None

    score = max(-15000, min(15000, score))
    return fen, score, depth, knodes


def process_evals(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    max_rows: int | None,
    shuffle: bool = True,
    seed: int = 42,
    batch_size: int = 1000000,
) -> int:
    """
    Process evaluation data, streaming to parquet in batches.

    Returns the number of rows written.
    """
    print(f"Loading {input_path}...")
    lf = pl_scan_ndjson_zstd(input_path)

    # Pre-count total rows for progress bar (skip if max_rows is set to avoid double scan)
    total_rows = max_rows
    if total_rows is None:
        print("Counting rows...")
        total_rows = len(lf)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp directory for batch parquet files
    tmp_dir = tempfile.mkdtemp(prefix="evals_")

    print("Processing evaluations...")
    count = 0
    written = 0
    batch_num = 0
    batch: list[tuple[str, int, int, int]] = []  # FEN strings, not compressed

    def flush_batch():
        nonlocal batch, batch_num
        if not batch:
            return
        # Batch compress all FENs at once
        fens = [row[0] for row in batch]
        compressed_fens = dummy_chess.compress_fens_batch(fens)
        compressed_batch = [
            (compressed_fens[i], batch[i][1], batch[i][2], batch[i][3])
            for i in range(len(batch))
        ]
        df = polars.DataFrame(
            compressed_batch,
            schema={
                "fen": polars.Binary,
                "score": polars.Int64,
                "depth": polars.Int64,
                "knodes": polars.Int64,
            },
            orient="row",
        )
        df.write_parquet(f"{tmp_dir}/batch_{batch_num:06d}.parquet")
        batch_num += 1
        batch = []

    pbar = tqdm.auto.tqdm(total=total_rows, desc="Evals", unit=" rows")
    for row in lf.iter_rows():
        if max_rows and count >= max_rows:
            break

        result = process_eval_row(row)
        if result:
            batch.append(result)
            written += 1

            if len(batch) >= batch_size:
                flush_batch()

        count += 1
        pbar.update(1)

    pbar.close()
    flush_batch()  # Write remaining

    print(f"Written {written} rows in {batch_num} batches")

    if batch_num == 0:
        print("No data to write")
        shutil.rmtree(tmp_dir)
        return 0

    # Combine batches and write to output
    print(f"Writing to {output_path}...")
    lf = polars.scan_parquet(f"{tmp_dir}/*.parquet")

    out_str = str(output_path)
    if out_str.endswith(".parquet") or not any(
        out_str.endswith(ext) for ext in [".csv", ".csv.gz", ".csv.zst"]
    ):
        # Stream directly to parquet without loading into memory
        lf.sink_parquet(output_path)
    else:
        # CSV requires collect (no sink_csv for LazyFrame with glob)
        combined = lf.collect(engine="streaming")
        if shuffle:
            print("Shuffling...")
            combined = combined.sample(fraction=1.0, shuffle=True, seed=seed)
        if out_str.endswith(".csv.gz"):
            combined.write_csv(output_path, compression="gzip")
        elif out_str.endswith(".csv.zst"):
            combined.write_csv(output_path, compression="zstd")
        else:
            combined.write_csv(output_path)

    # Cleanup
    shutil.rmtree(tmp_dir)

    print(f"Saved {written} rows -> {output_path}")
    return written


# =============================================================================
# Endgame generation
# =============================================================================

DEFAULT_TABLEBASE_PATH = str(
    pathlib.Path(__file__).parent.parent / "external" / "syzygy" / "src"
)

PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
PIECES_NO_PAWN = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


@dataclasses.dataclass
class EndgameConfig:
    """Configuration for endgame generation."""

    min_pieces: int = 3
    max_pieces: int | None = None
    include_pawns: bool = True

    def resolve_max_pieces(self, tablebase_path: str) -> int:
        if self.max_pieces is not None:
            return self.max_pieces
        return detect_max_pieces(tablebase_path)


def detect_max_pieces(tablebase_path: str) -> int:
    """Detect maximum piece count supported by available tablebases."""
    tb_path = pathlib.Path(tablebase_path)
    if not tb_path.exists():
        return 3

    max_pieces = 3
    for rtbw_file in tb_path.glob("*.rtbw"):
        name = rtbw_file.stem
        piece_count = sum(1 for c in name if c in "KQRBNPkqrbnp")
        max_pieces = max(max_pieces, piece_count)

    return max_pieces


def random_square(
    rng: random.Random, exclude: set[int] | None = None, pawn: bool = False
) -> int:
    """Generate a random square."""
    if exclude is None:
        exclude = set()

    if pawn:
        valid = [sq for sq in range(8, 56) if sq not in exclude]
    else:
        valid = [sq for sq in range(64) if sq not in exclude]

    return rng.choice(valid) if valid else -1


def generate_random_position(
    config: EndgameConfig, max_pieces: int, rng: random.Random | None = None
) -> chess.Board | None:
    """Generate a random endgame position."""
    if rng is None:
        rng = random.Random()

    n_pieces = rng.randint(config.min_pieces, max_pieces)
    n_extra = n_pieces - 2

    board = chess.Board.empty()

    # Place kings
    occupied: set[int] = set()
    wk_sq = random_square(rng, occupied)
    occupied.add(wk_sq)

    adjacent = set()
    for d in [-9, -8, -7, -1, 1, 7, 8, 9]:
        adj = wk_sq + d
        if 0 <= adj < 64:
            wk_file = wk_sq % 8
            adj_file = adj % 8
            if abs(wk_file - adj_file) <= 1:
                adjacent.add(adj)

    bk_sq = random_square(rng, occupied | adjacent)
    if bk_sq == -1:
        return None
    occupied.add(bk_sq)

    board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))

    # Place extra pieces
    for _ in range(n_extra):
        color = rng.choice([True, False])
        if config.include_pawns:
            piece_type = rng.choice(PIECES)
        else:
            piece_type = rng.choice(PIECES_NO_PAWN)

        is_pawn = piece_type == chess.PAWN
        sq = random_square(rng, occupied, pawn=is_pawn)
        if sq == -1:
            return None
        occupied.add(sq)
        board.set_piece_at(sq, chess.Piece(piece_type, color))

    board.turn = rng.choice([chess.WHITE, chess.BLACK])
    board.castling_rights = chess.BB_EMPTY

    if not board.is_valid():
        return None

    board.turn = not board.turn
    if board.is_check():
        return None
    board.turn = not board.turn

    return board


def dtz_to_score(dtz: int, wdl: int, turn: bool) -> float:
    """Convert tablebase DTZ to centipawn score."""
    if wdl == 0:
        return 0.0

    if wdl == 1:
        score = 90.0
    elif wdl == -1:
        score = -90.0
    elif wdl == 2:
        abs_dtz = abs(dtz)
        clamped_dtz = min(max(abs_dtz, 1), 100)
        score = 1000.0 - (clamped_dtz - 1) * 2.0
    elif wdl == -2:
        abs_dtz = abs(dtz)
        clamped_dtz = min(max(abs_dtz, 1), 100)
        score = -1000.0 + (clamped_dtz - 1) * 2.0
    else:
        score = 0.0

    if not turn:
        score = -score

    return score


def wdl_to_score(wdl: int, turn: bool) -> float:
    """Convert tablebase WDL to centipawn score."""
    wdl_scores = {2: 1000.0, 1: 90.0, 0: 0.0, -1: -90.0, -2: -1000.0}
    score = wdl_scores.get(wdl, 0.0)
    if not turn:
        score = -score
    return score


# Tablebase = perfect information, use high depth marker
TABLEBASE_DEPTH = 255


def process_endgames(
    n_positions: int,
    tablebase_path: str | None = None,
    min_pieces: int = 3,
    max_pieces: int | None = None,
    include_pawns: bool = True,
    use_dtz: bool = True,
) -> list[tuple[bytes, int, int, int]]:
    """Generate endgame positions with tablebase evaluations."""
    if tablebase_path is None:
        tablebase_path = DEFAULT_TABLEBASE_PATH

    config = EndgameConfig(min_pieces, max_pieces, include_pawns)
    resolved_max_pieces = config.resolve_max_pieces(tablebase_path)

    print(f"Generating {n_positions} endgames...")
    print(f"  Tablebase: {tablebase_path}")
    print(f"  Pieces: {min_pieces}-{resolved_max_pieces}")
    print(f"  Pawns: {'yes' if include_pawns else 'no'}")
    print(f"  DTZ scoring: {'yes' if use_dtz else 'no'}")

    tablebase = chess.syzygy.Tablebase()
    tablebase.add_directory(tablebase_path)

    # Collect FEN strings first, then batch compress
    fen_results: list[tuple[str, int, int, int]] = []
    attempts = 0
    max_attempts = n_positions * 1000

    with tqdm.auto.tqdm(total=n_positions, desc="Endgames") as pbar:
        while len(fen_results) < n_positions and attempts < max_attempts:
            attempts += 1

            board = generate_random_position(config, resolved_max_pieces)
            if board is None:
                continue

            try:
                wdl = tablebase.probe_wdl(board)
                if wdl is None:
                    continue

                if use_dtz:
                    dtz = tablebase.probe_dtz(board)
                    if dtz is None:
                        score = wdl_to_score(wdl, board.turn)
                    else:
                        score = dtz_to_score(dtz, wdl, board.turn)
                else:
                    score = wdl_to_score(wdl, board.turn)

            except chess.syzygy.MissingTableError:
                continue

            fen_results.append((board.fen(), int(score), TABLEBASE_DEPTH, 0))
            pbar.update(1)

    tablebase.close()

    # Batch compress all FENs
    if fen_results:
        fens = [row[0] for row in fen_results]
        compressed_fens = dummy_chess.compress_fens_batch(fens)
        results: list[tuple[bytes, int, int, int]] = [
            (
                compressed_fens[i],
                fen_results[i][1],
                fen_results[i][2],
                fen_results[i][3],
            )
            for i in range(len(fen_results))
        ]
        return results
    return []


# =============================================================================
# Parquet shuffling (external sort)
# =============================================================================


def dedupe_parquet(
    input_path: str,
    output_path: str | None = None,
    num_buckets: int = 256,
    row_group_size: int = 100_000,
) -> tuple[int, int, int]:
    """
    Deduplicate a parquet file based on the 'fen' column.

    Uses external sort approach:
    1. Hash each FEN, distribute rows to buckets by hash prefix
    2. For each bucket: load, sort by hash, dedupe, write unique rows

    This keeps memory bounded to ~(total_rows / num_buckets) * row_size.

    Args:
        input_path: Input parquet file
        output_path: Output parquet file (None = stats only, no output)
        num_buckets: Number of temp buckets (more = less RAM per bucket)
        row_group_size: Rows per output row group

    Returns:
        Tuple of (total_rows, unique_rows, duplicates_removed)

    RAM usage: ~(total_rows / num_buckets) * row_size
               With 256 buckets on 354M rows, ~1.4M rows per bucket
    """
    import hashlib

    input_path = pathlib.Path(input_path)
    stats_only = output_path is None
    if not stats_only:
        output_path = pathlib.Path(output_path)

    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="dedupe_"))

    try:
        # Phase 1: Distribute rows to buckets by FEN hash
        print(f"Phase 1: Distributing to {num_buckets} buckets...")

        pf = pyarrow.parquet.ParquetFile(input_path)
        total_rows = pf.metadata.num_rows
        num_row_groups = pf.metadata.num_row_groups

        # Bucket buffers
        bucket_buffers: list[list[tuple[bytes, dict]]] = [
            [] for _ in range(num_buckets)
        ]
        buffer_limit = max(1000, row_group_size // num_buckets)

        def flush_bucket(bucket_idx: int):
            if not bucket_buffers[bucket_idx]:
                return

            rows = [row for _, row in bucket_buffers[bucket_idx]]
            hashes = [h for h, _ in bucket_buffers[bucket_idx]]

            df = polars.DataFrame(rows)
            df = df.with_columns(polars.Series("_hash", hashes))

            bucket_file = tmp_dir / f"bucket_{bucket_idx:04d}.parquet"
            if bucket_file.exists():
                existing = polars.read_parquet(bucket_file)
                df = polars.concat([existing, df])

            df.write_parquet(bucket_file)
            bucket_buffers[bucket_idx] = []

        with tqdm.auto.tqdm(
            total=total_rows, desc="Distributing", unit=" rows"
        ) as pbar:
            for rg_idx in range(num_row_groups):
                table = pf.read_row_group(rg_idx)
                df = polars.from_arrow(table)

                for row in df.iter_rows(named=True):
                    fen_bytes = (
                        row["fen"]
                        if isinstance(row["fen"], bytes)
                        else row["fen"].encode()
                    )
                    h = hashlib.md5(fen_bytes).digest()

                    bucket_idx = h[0] % num_buckets
                    bucket_buffers[bucket_idx].append((h, row))

                    if len(bucket_buffers[bucket_idx]) >= buffer_limit:
                        flush_bucket(bucket_idx)

                pbar.update(len(df))

        for bucket_idx in range(num_buckets):
            flush_bucket(bucket_idx)

        # Phase 2: Dedupe each bucket, write to output (or just count if stats_only)
        print("Phase 2: Deduplicating buckets...")

        bucket_files = sorted(tmp_dir.glob("bucket_*.parquet"))

        if not bucket_files:
            print("No data to dedupe")
            return 0, 0, 0

        unique_rows = 0

        if stats_only:
            # Just count unique hashes, don't write output
            with tqdm.auto.tqdm(
                total=len(bucket_files), desc="Counting", unit=" buckets"
            ) as pbar:
                for bucket_file in bucket_files:
                    df = polars.read_parquet(bucket_file)
                    unique_in_bucket = df["_hash"].n_unique()
                    unique_rows += unique_in_bucket
                    pbar.update(1)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_buffer: list[dict] = []

            first_bucket = polars.read_parquet(bucket_files[0]).drop("_hash")
            arrow_schema = first_bucket.to_arrow().schema

            with pyarrow.parquet.ParquetWriter(
                output_path, arrow_schema, compression="zstd"
            ) as writer:

                def flush_output():
                    nonlocal output_buffer, unique_rows
                    if not output_buffer:
                        return
                    df = polars.DataFrame(output_buffer)
                    writer.write_table(df.to_arrow())
                    unique_rows += len(output_buffer)
                    output_buffer = []

                with tqdm.auto.tqdm(
                    total=len(bucket_files), desc="Deduping", unit=" buckets"
                ) as pbar:
                    for bucket_file in bucket_files:
                        df = polars.read_parquet(bucket_file)

                        # Sort by hash, then dedupe by hash (keeps first)
                        df = df.sort("_hash")

                        seen_hashes: set[bytes] = set()
                        for row in df.iter_rows(named=True):
                            h = row["_hash"]
                            if h not in seen_hashes:
                                seen_hashes.add(h)
                                row_copy = {
                                    k: v for k, v in row.items() if k != "_hash"
                                }
                                output_buffer.append(row_copy)

                                if len(output_buffer) >= row_group_size:
                                    flush_output()

                        pbar.update(1)

                flush_output()

        duplicates = total_rows - unique_rows
        print(f"Total: {total_rows}, Unique: {unique_rows}, Duplicates: {duplicates}")
        return total_rows, unique_rows, duplicates

    finally:
        shutil.rmtree(tmp_dir)


def shuffle_parquet(
    input_path: str,
    output_path: str,
    num_buckets: int = 64,
    row_group_size: int = 100_000,
    seed: int | None = None,
) -> int:
    """
    Shuffle a parquet file with minimal RAM using two-phase external sort.

    Phase 1: Read input row-group by row-group, assign random keys,
             distribute rows to N bucket files based on key range.
    Phase 2: Read each bucket, sort by key, write to output.

    This achieves a true shuffle where any input row can end up anywhere
    in the output, without loading the entire dataset into memory.

    Args:
        input_path: Input parquet file
        output_path: Output parquet file (will be overwritten)
        num_buckets: Number of temporary bucket files (more = less RAM per bucket)
        row_group_size: Rows per output row group
        seed: Random seed for reproducibility

    Returns:
        Number of rows written

    RAM usage: ~(largest_bucket_size / num_buckets) * row_size
               With 64 buckets, expect ~1.5% of total data in RAM at peak
    """
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)

    rng = random.Random(seed)

    # Create temp directory for buckets
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="shuffle_"))

    try:
        # Phase 1: Distribute rows to buckets based on random key
        print(f"Phase 1: Distributing to {num_buckets} buckets...")

        pf = pyarrow.parquet.ParquetFile(input_path)
        total_rows = pf.metadata.num_rows
        num_row_groups = pf.metadata.num_row_groups
        schema = pf.schema_arrow

        # Initialize bucket buffers and files
        bucket_buffers: list[list[tuple[float, dict]]] = [
            [] for _ in range(num_buckets)
        ]
        bucket_counts = [0] * num_buckets
        buffer_limit = max(
            1000, row_group_size // num_buckets
        )  # Flush threshold per bucket

        def flush_bucket(bucket_idx: int):
            """Write bucket buffer to its temp file."""
            if not bucket_buffers[bucket_idx]:
                return

            # Sort by key within this chunk
            bucket_buffers[bucket_idx].sort(key=lambda x: x[0])

            # Extract rows (without keys)
            rows = [row for _, row in bucket_buffers[bucket_idx]]

            df = polars.DataFrame(rows)

            # Append to bucket file
            bucket_file = tmp_dir / f"bucket_{bucket_idx:04d}.parquet"
            if bucket_file.exists():
                # Read existing, concatenate, rewrite (not ideal but simple)
                existing = polars.read_parquet(bucket_file)
                df = polars.concat([existing, df])

            df.write_parquet(bucket_file)
            bucket_counts[bucket_idx] += len(bucket_buffers[bucket_idx])
            bucket_buffers[bucket_idx] = []

        # Process input row groups
        with tqdm.auto.tqdm(
            total=total_rows, desc="Distributing", unit=" rows"
        ) as pbar:
            for rg_idx in range(num_row_groups):
                table = pf.read_row_group(rg_idx)
                df = polars.from_arrow(table)

                for row in df.iter_rows(named=True):
                    # Assign random key
                    key = rng.random()

                    # Determine bucket (uniform distribution)
                    bucket_idx = int(key * num_buckets)
                    bucket_idx = min(bucket_idx, num_buckets - 1)  # Handle key == 1.0

                    bucket_buffers[bucket_idx].append((key, row))

                    # Flush if buffer full
                    if len(bucket_buffers[bucket_idx]) >= buffer_limit:
                        flush_bucket(bucket_idx)

                pbar.update(len(df))

        # Flush remaining buffers
        for bucket_idx in range(num_buckets):
            flush_bucket(bucket_idx)

        print(
            f"  Bucket sizes: min={min(bucket_counts)}, max={max(bucket_counts)}, "
            f"avg={sum(bucket_counts) // num_buckets}"
        )

        # Phase 2: Read buckets in order, sort each, write to output
        print("Phase 2: Merging buckets...")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect all bucket files in order
        bucket_files = sorted(tmp_dir.glob("bucket_*.parquet"))

        if not bucket_files:
            print("No data to shuffle")
            return 0

        # Use sink to stream output
        written = 0
        output_buffer: list[dict] = []

        def flush_output(writer):
            nonlocal output_buffer, written
            if not output_buffer:
                return

            df = polars.DataFrame(output_buffer)
            # Convert to arrow and write
            writer.write_table(df.to_arrow())
            written += len(output_buffer)
            output_buffer = []

        # Create output writer
        first_bucket = polars.read_parquet(bucket_files[0])
        arrow_schema = first_bucket.to_arrow().schema

        with pyarrow.parquet.ParquetWriter(
            output_path, arrow_schema, compression="zstd"
        ) as writer:
            with tqdm.auto.tqdm(total=total_rows, desc="Writing", unit=" rows") as pbar:
                for bucket_file in bucket_files:
                    # Read bucket
                    df = polars.read_parquet(bucket_file)

                    if "_sort_key" in df.columns:
                        df = df.drop("_sort_key")

                    # Add to output buffer
                    for row in df.iter_rows(named=True):
                        output_buffer.append(row)

                        if len(output_buffer) >= row_group_size:
                            flush_output(writer)

                    pbar.update(len(df))

            # Flush remaining
            flush_output(writer)

        print(f"Shuffled {written} rows -> {output_path}")
        return written

    finally:
        # Cleanup temp directory
        shutil.rmtree(tmp_dir)


# =============================================================================
# Output
# =============================================================================


def save_data(
    data: list[tuple[bytes, int, int, int]],
    output_path: pathlib.Path,
    shuffle: bool = True,
    seed: int = 42,
):
    """Save data to parquet or CSV file with columns: fen (binary), score, depth, knodes."""
    if shuffle:
        random.seed(seed)
        random.shuffle(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_df = polars.DataFrame(
        data,
        schema={
            "fen": polars.Binary,
            "score": polars.Int64,
            "depth": polars.Int64,
            "knodes": polars.Int64,
        },
        orient="row",
    )

    out_str = str(output_path)
    if out_str.endswith(".parquet"):
        out_df.write_parquet(output_path)
    elif out_str.endswith(".csv.gz"):
        out_df.write_csv(output_path, compression="gzip")
    elif out_str.endswith(".csv.zst"):
        out_df.write_csv(output_path, compression="zstd")
    elif out_str.endswith(".csv"):
        out_df.write_csv(output_path)
    else:
        # Default to parquet
        out_df.write_parquet(output_path)

    print(f"Saved {len(out_df)} rows -> {output_path}")

    if data:
        all_scores = [s for _, s, _, _ in data]
        all_depths = [d for _, _, d, _ in data]
        print(
            f"Score stats: min={min(all_scores)}, max={max(all_scores)}, "
            f"mean={sum(all_scores) / len(all_scores):.0f}"
        )
        print(
            f"Depth stats: min={min(all_depths)}, max={max(all_depths)}, "
            f"mean={sum(all_depths) / len(all_depths):.0f}"
        )


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Preprocess data for NNUE training")
    parser.add_argument(
        "source",
        choices=["puzzles", "evals", "endgames", "shuffle", "dedupe"],
        help="Data source: 'puzzles', 'evals', 'endgames', 'shuffle', or 'dedupe'",
    )
    parser.add_argument("-i", "--input", default=None, help="Input file")
    parser.add_argument("-o", "--output", default=None, help="Output file")
    parser.add_argument(
        "-n", "--max", type=int, default=None, help="Max rows to process/generate"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--toy", action="store_true", help="Generate toy dataset")

    # Puzzle-specific
    parser.add_argument("-e", "--engine", default=None, help="Path to UCI engine")
    parser.add_argument("-d", "--depth", type=int, default=12, help="UCI engine depth")

    # Endgame-specific
    parser.add_argument(
        "-t", "--tablebase", default=None, help="Path to Syzygy tablebases"
    )
    parser.add_argument("--min-pieces", type=int, default=3)
    parser.add_argument("--max-pieces", type=int, default=None)
    parser.add_argument("--no-pawns", action="store_true")
    parser.add_argument(
        "--no-dtz", action="store_true", help="Use WDL instead of DTZ scoring"
    )

    # Shuffle/dedupe-specific
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=64,
        help="Number of temp buckets for shuffle/dedupe (more = less RAM)",
    )
    parser.add_argument(
        "--row-group-size", type=int, default=100_000, help="Rows per output row group"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="For dedupe: only count duplicates, don't write output",
    )

    args = parser.parse_args()

    # Set defaults
    if args.input is None:
        if args.source == "puzzles":
            args.input = "data/lichess_db_puzzle.csv.zst"
        elif args.source == "evals":
            args.input = "data/lichess_db_eval.jsonl.zst"

    if args.output is None:
        args.output = f"data/{args.source}.parquet"

    if args.toy:
        if args.source == "puzzles":
            args.max = 35000
        elif args.source == "evals":
            args.max = 10000
        elif args.source == "endgames":
            args.max = 10000
        output_path = pathlib.Path(args.output)
        args.output = str(output_path.parent / f"toy_{output_path.name}")

    # Process
    data: list[tuple[bytes, int, int, int]] = []

    if args.source == "puzzles":
        input_path = pathlib.Path(args.input)
        if not input_path.exists():
            print(f"Error: {args.input} not found")
            print(
                "Download: wget https://database.lichess.org/lichess_db_puzzle.csv.zst"
            )
            sys.exit(1)
        if not args.engine:
            print("Error: --engine is required for puzzle preprocessing")
            sys.exit(1)
        # Puzzles handles its own output (streaming)
        process_puzzles(
            input_path,
            pathlib.Path(args.output),
            args.max,
            args.engine,
            args.depth,
            args.tablebase,
        )
        return

    elif args.source == "evals":
        input_path = pathlib.Path(args.input)
        if not input_path.exists():
            print(f"Error: {args.input} not found")
            print(
                "Download: wget https://database.lichess.org/lichess_db_eval.jsonl.zst"
            )
            sys.exit(1)
        # Evals handles its own output (streaming)
        process_evals(
            input_path,
            pathlib.Path(args.output),
            args.max,
            shuffle=not args.no_shuffle,
            seed=args.seed,
        )
        return

    elif args.source == "endgames":
        if args.max is None:
            args.max = 100000
        data = process_endgames(
            args.max,
            args.tablebase,
            args.min_pieces,
            args.max_pieces,
            not args.no_pawns,
            not args.no_dtz,
        )

        print(f"Valid: {len(data)}")

        save_data(
            data,
            pathlib.Path(args.output),
            shuffle=not args.no_shuffle,
            seed=args.seed,
        )
        return

    elif args.source == "shuffle":
        if args.input is None:
            print("Error: --input is required for shuffle")
            sys.exit(1)
        input_path = pathlib.Path(args.input)
        if not input_path.exists():
            print(f"Error: {args.input} not found")
            sys.exit(1)
        if args.output is None:
            # Default: input_shuffled.parquet
            args.output = str(input_path.parent / f"{input_path.stem}_shuffled.parquet")

        shuffle_parquet(
            input_path,
            pathlib.Path(args.output),
            num_buckets=args.num_buckets,
            row_group_size=args.row_group_size,
            seed=args.seed,
        )
        return

    elif args.source == "dedupe":
        if args.input is None:
            print("Error: --input is required for dedupe")
            sys.exit(1)
        input_path = pathlib.Path(args.input)
        if not input_path.exists():
            print(f"Error: {args.input} not found")
            sys.exit(1)

        if args.stats_only:
            output = None
        elif args.output is None:
            output = input_path.parent / f"{input_path.stem}_deduped.parquet"
        else:
            output = pathlib.Path(args.output)

        dedupe_parquet(
            input_path,
            output,
            num_buckets=args.num_buckets,
            row_group_size=args.row_group_size,
        )
        return


if __name__ == "__main__":
    main()
