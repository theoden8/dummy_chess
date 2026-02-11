#!/usr/bin/env python3
"""
Preprocess data for NNUE training.

Supports five data sources:
1. puzzles: lichess_db_puzzle.csv.zst - uses UCI engine for evaluation
2. evals: lichess_db_eval.jsonl.zst - uses pre-computed engine evals from lichess
3. endgames: generates random endgame positions using Syzygy tablebases
4. games: lichess game archives (.pgn.zst) - extracts random positions with engine eval
5. games-all: stream and process all non-blocklisted months from Lichess

Also provides utilities:
- shuffle: memory-efficient parquet shuffling
- dedupe: memory-efficient deduplication

Outputs parquet files with columns: fen (binary), score, depth, knodes

Usage for games-all (streaming all months):
    uv run python preprocess.py games-all --use-pgn-evals --after 2020-01
    uv run python preprocess.py games-all --use-pgn-evals -n 100000
    uv run python preprocess.py games-all -e stockfish --after 2020-01  # with engine
    uv run python preprocess.py games-all --list-months

STYLE: Do NOT use `from X import Y` - use fully qualified names (e.g. pathlib.Path, not Path)
STYLE: Do NOT use import aliases (e.g. `import numpy as np`) - use full module names
"""

import argparse
import dataclasses
import hashlib
import io
import json
import itertools
import pathlib
import random
import shutil
import sys
import tempfile
import typing
import urllib.error
import urllib.request

import chess
import chess.engine
import chess.pgn
import chess.syzygy
import dummy_chess
import orjson
import polars
import pyarrow.parquet
import tqdm.auto
import zstandard

import download


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

    # Create temp directory for batch parquet files NEXT TO output (not in /tmp)
    tmp_dir = output_path.parent / f".{output_path.stem}_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

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

    # Create temp directory for batch parquet files NEXT TO output (not in /tmp)
    tmp_dir = output_path.parent / f".{output_path.stem}_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

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
        # use_pyarrow=True avoids polars dictionary encoding issues
        df.write_parquet(f"{tmp_dir}/batch_{batch_num:06d}.parquet", use_pyarrow=True)
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
    print(f"Combining {batch_num} batch files...")
    batch_files = sorted(tmp_dir.glob("batch_*.parquet"))

    out_str = str(output_path)
    if out_str.endswith(".parquet") or not any(
        out_str.endswith(ext) for ext in [".csv", ".csv.gz", ".csv.zst"]
    ):
        # Stream batches to output parquet file (memory efficient)
        first_table = pyarrow.parquet.read_table(batch_files[0])
        with pyarrow.parquet.ParquetWriter(
            output_path, first_table.schema, compression="zstd"
        ) as writer:
            writer.write_table(first_table)
            del first_table
            for batch_file in tqdm.auto.tqdm(
                batch_files[1:], desc="Writing", unit=" batches"
            ):
                table = pyarrow.parquet.read_table(batch_file)
                writer.write_table(table)
                del table
    else:
        # CSV output (requires loading into memory for shuffle)
        combined = polars.concat([polars.read_parquet(f) for f in batch_files])
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
    engine_path: str | None = None,
    engine_depth: int = 12,
) -> list[tuple[bytes, int, int, int]]:
    """
    Generate endgame positions with evaluations.

    Scoring priority:
    1. If engine_path provided: use UCI engine evaluation
    2. If use_dtz=True: use DTZ tablebase scores (falls back to WDL if DTZ unavailable)
    3. Otherwise: use WDL tablebase scores

    Tablebase is always used to filter positions (only include positions with valid WDL).
    """
    if tablebase_path is None:
        tablebase_path = DEFAULT_TABLEBASE_PATH

    config = EndgameConfig(min_pieces, max_pieces, include_pawns)
    resolved_max_pieces = config.resolve_max_pieces(tablebase_path)

    print(f"Generating {n_positions} endgames...")
    print(f"  Tablebase: {tablebase_path}")
    print(f"  Pieces: {min_pieces}-{resolved_max_pieces}")
    print(f"  Pawns: {'yes' if include_pawns else 'no'}")
    if engine_path:
        print(f"  Scoring: UCI engine ({engine_path}, depth={engine_depth})")
    elif use_dtz:
        print(f"  Scoring: DTZ tablebase (fallback to WDL)")
    else:
        print(f"  Scoring: WDL tablebase")

    tablebase = chess.syzygy.Tablebase()
    tablebase.add_directory(tablebase_path)

    # Start engine if provided
    engine = None
    if engine_path:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        # Configure engine with tablebase path for better endgame play
        try:
            engine.configure({"SyzygyPath": tablebase_path})
        except chess.engine.EngineError:
            pass  # Engine may not support Syzygy

    # Collect FEN strings first, then batch compress
    fen_results: list[tuple[str, int, int, int]] = []
    attempts = 0
    max_attempts = n_positions * 1000

    try:
        with tqdm.auto.tqdm(total=n_positions, desc="Endgames") as pbar:
            while len(fen_results) < n_positions and attempts < max_attempts:
                attempts += 1

                board = generate_random_position(config, resolved_max_pieces)
                if board is None:
                    continue

                try:
                    # Always check tablebase WDL to filter valid positions
                    wdl = tablebase.probe_wdl(board)
                    if wdl is None:
                        continue

                    # Get score based on configured method
                    if engine:
                        # Use UCI engine for scoring
                        score, depth, knodes = uci_eval(board, engine, engine_depth)
                    elif use_dtz:
                        dtz = tablebase.probe_dtz(board)
                        if dtz is None:
                            score = wdl_to_score(wdl, board.turn)
                        else:
                            score = dtz_to_score(dtz, wdl, board.turn)
                        depth = TABLEBASE_DEPTH
                        knodes = 0
                    else:
                        score = wdl_to_score(wdl, board.turn)
                        depth = TABLEBASE_DEPTH
                        knodes = 0

                except chess.syzygy.MissingTableError:
                    continue

                fen_results.append((board.fen(), int(score), depth, knodes))
                pbar.update(1)

    finally:
        tablebase.close()
        if engine:
            engine.quit()

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
# Game position extraction (from Lichess PGN archives)
# =============================================================================

# Months with known data issues - skip entirely
BLOCKLISTED_MONTHS = frozenset(
    [
        "2021-03",  # Datacenter fire, incorrect results
    ]
)

# Base URLs for game archives
LICHESS_STANDARD_URL = (
    "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
)
LICHESS_CHESS960_URL = (
    "https://database.lichess.org/chess960/lichess_db_chess960_rated_{month}.pgn.zst"
)


@dataclasses.dataclass
class GameExtractionConfig:
    """Configuration for game position extraction."""

    # Position selection
    min_ply: int = 10  # Skip opening positions
    end_margin: int = 5  # Skip positions near game end
    min_pieces: int = 7  # Skip simple endgames (use tablebases instead)

    # Game filtering
    min_elo: int | None = None  # Optional rating floor
    max_elo: int | None = None  # Optional rating ceiling
    min_game_length: int = 20  # Minimum plies in game
    skip_bots: bool = True  # Skip games with BOT players

    # Engine settings
    depth: int = 8  # Stockfish search depth

    # Processing
    batch_size: int = 10000  # Positions per parquet row group


@dataclasses.dataclass
class GameExtractionProgress:
    """Checkpoint for resumable game extraction."""

    url_or_path: str  # URL or file path being processed
    games_processed: int = 0
    positions_extracted: int = 0
    bytes_read: int = 0  # For HTTP resume

    def save(self, path: pathlib.Path) -> None:
        """Save progress to JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: pathlib.Path) -> "GameExtractionProgress":
        """Load progress from JSON file."""
        import json

        with open(path) as f:
            data = json.load(f)
        # Handle old format
        if "month" in data:
            data["url_or_path"] = data.pop("month", "")
            data.pop("variant", None)
        return cls(**data)


def should_process_game(game: chess.pgn.Game, config: GameExtractionConfig) -> bool:
    """Check if game should be processed based on headers."""
    headers = game.headers

    # Skip bot games
    if config.skip_bots:
        if headers.get("WhiteTitle") == "BOT" or headers.get("BlackTitle") == "BOT":
            return False

    # Check ratings
    try:
        white_elo = int(headers.get("WhiteElo", "0"))
        black_elo = int(headers.get("BlackElo", "0"))
    except ValueError:
        white_elo = black_elo = 0

    if config.min_elo is not None:
        if white_elo < config.min_elo or black_elo < config.min_elo:
            return False

    if config.max_elo is not None:
        if white_elo > config.max_elo or black_elo > config.max_elo:
            return False

    return True


def parse_eval_comment(comment: str) -> int | None:
    """
    Parse [%eval X] from PGN comment.

    Returns centipawn score (from white's perspective) or None if not found.
    Mate scores are converted to +/- 10000.
    """
    import re

    match = re.search(r"\[%eval\s+([^\]]+)\]", comment)
    if not match:
        return None

    eval_str = match.group(1).strip()

    # Handle mate scores like "#5" or "#-3"
    if eval_str.startswith("#"):
        mate_in = eval_str[1:]
        try:
            mate_plies = int(mate_in)
            return 10000 if mate_plies > 0 else -10000
        except ValueError:
            return None

    # Handle centipawn scores
    try:
        # Lichess evals are in pawns (e.g., "1.23"), convert to centipawns
        score = float(eval_str)
        return int(score * 100)
    except ValueError:
        return None


def extract_position_from_game(
    game: chess.pgn.Game,
    config: GameExtractionConfig,
    rng: random.Random,
) -> tuple[str, int] | None:
    """
    Extract one random position from a game (for engine evaluation).

    Returns (fen, ply_index) or None if no valid position found.
    """
    # Collect valid positions (ply_index, fen)
    valid_positions: list[tuple[int, str]] = []

    board = game.board()  # Handles Chess960 starting position automatically
    ply = 0
    total_plies = len(list(game.mainline_moves()))

    # Check minimum game length
    if total_plies < config.min_game_length:
        return None

    for node in game.mainline():
        # Get position BEFORE this move
        piece_count = len(board.piece_map())

        if (
            ply >= config.min_ply
            and ply < total_plies - config.end_margin
            and piece_count >= config.min_pieces
            and not board.is_game_over()
        ):
            valid_positions.append((ply, board.fen()))

        # Make the move
        board.push(node.move)
        ply += 1

    if not valid_positions:
        return None

    # Pick random position
    ply_idx, fen = rng.choice(valid_positions)
    return fen, ply_idx


def extract_position_with_eval(
    game: chess.pgn.Game,
    config: GameExtractionConfig,
    rng: random.Random,
) -> tuple[str, int, int] | None:
    """
    Extract one random position with its PGN eval annotation.

    Returns (fen, score_cp, ply_index) or None if no valid position found.
    Only considers positions that have [%eval] annotations.
    """
    # Collect valid positions (ply_index, fen, score)
    valid_positions: list[tuple[int, str, int]] = []

    board = game.board()  # Handles Chess960 starting position automatically
    ply = 0
    total_plies = len(list(game.mainline_moves()))

    # Check minimum game length
    if total_plies < config.min_game_length:
        return None

    for node in game.mainline():
        # Parse eval from comment
        comment = node.comment or ""
        score = parse_eval_comment(comment)

        # Get position BEFORE this move
        piece_count = len(board.piece_map())

        if (
            score is not None
            and ply >= config.min_ply
            and ply < total_plies - config.end_margin
            and piece_count >= config.min_pieces
            and not board.is_game_over()
        ):
            valid_positions.append((ply, board.fen(), score))

        # Make the move
        board.push(node.move)
        ply += 1

    if not valid_positions:
        return None

    # Pick random position
    ply_idx, fen, score = rng.choice(valid_positions)
    return fen, score, ply_idx


def iter_pgn_text(stream: typing.IO[str]) -> typing.Iterator[str]:
    """
    Iterate over raw PGN game text from a stream.

    Fast text-based splitting - doesn't parse the PGN, just finds game boundaries.
    Yields complete PGN game strings (headers + moves).
    """
    lines: list[str] = []
    in_game = False

    for line in stream:
        stripped = line.strip()

        # Empty line after moves signals end of game
        if not stripped and in_game and lines:
            # Check if last non-empty line looks like it ends a game
            # (result like 1-0, 0-1, 1/2-1/2, or *)
            last_content = ""
            for prev_line in reversed(lines):
                if prev_line.strip():
                    last_content = prev_line.strip()
                    break

            if any(last_content.endswith(r) for r in ["1-0", "0-1", "1/2-1/2", "*"]):
                yield "".join(lines)
                lines = []
                in_game = False
                continue

        # Header line starts a new game
        if stripped.startswith("["):
            if not in_game:
                in_game = True
        elif stripped and in_game and not stripped.startswith("["):
            # Non-header content (moves)
            pass

        if stripped or in_game:
            lines.append(line)

    # Yield final game if any
    if lines:
        yield "".join(lines)


def parse_pgn_headers(pgn_text: str) -> dict[str, str]:
    """
    Fast header extraction from PGN text.
    Returns dict of header name -> value.
    """
    headers: dict[str, str] = {}
    for line in pgn_text.split("\n"):
        line = line.strip()
        if not line.startswith("["):
            if line and not line[0].isdigit():
                break  # End of headers (moves start)
            continue
        # Parse [Tag "Value"]
        if line.endswith("]"):
            content = line[1:-1]
            space_idx = content.find(" ")
            if space_idx > 0:
                tag = content[:space_idx]
                value = content[space_idx + 1 :].strip()
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                headers[tag] = value
    return headers


def iter_pgn_games(stream: typing.IO[str]) -> typing.Iterator[chess.pgn.Game]:
    """
    Iterate over PGN games from a text stream using python-chess.

    Yields chess.pgn.Game objects.
    """
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        yield game


def process_games_pgn_evals(
    source: str,  # URL or file path
    output_path: pathlib.Path,
    config: GameExtractionConfig,
    max_positions: int | None = None,
    seed: int = 42,
    resume: bool = False,
    use_torrent: bool = False,
    use_cpp_parser: bool = False,
    disable_progress: bool = False,
) -> int:
    """
    Process PGN game archive, extracting positions with their PGN [%eval] annotations.

    No engine needed - uses evals already in the PGN from Lichess analysis.

    Supports both HTTP URLs (streaming), BitTorrent (streaming), and local files.
    Uses batch files for atomic writes and full resumability.

    Args:
        source: URL (http/https) or local file path to .pgn.zst
        output_path: Output parquet file
        config: Extraction configuration
        max_positions: Maximum positions to extract (None = all)
        seed: Random seed
        resume: Whether to resume from checkpoint
        use_torrent: Use BitTorrent for streaming (appends .torrent to URL)

    Returns:
        Number of positions extracted
    """
    rng = random.Random(seed)
    is_url = source.startswith("http://") or source.startswith("https://")

    # Progress tracking
    progress_path = output_path.with_suffix(".progress.json")
    progress = GameExtractionProgress(url_or_path=source)

    if resume and progress_path.exists():
        progress = GameExtractionProgress.load(progress_path)
        if progress.url_or_path != source:
            print("Warning: Progress file is for different source, starting fresh")
            progress = GameExtractionProgress(url_or_path=source)
        else:
            print(
                f"Resuming from game {progress.games_processed}, "
                f"position {progress.positions_extracted}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Parquet schema - no knodes for PGN evals, use placeholder depth
    arrow_schema = pyarrow.schema(
        [
            ("fen", pyarrow.binary()),
            ("score", pyarrow.int64()),
            ("depth", pyarrow.int64()),
            ("knodes", pyarrow.int64()),
        ]
    )

    # Use temp directory for batch files (next to output, not /tmp)
    tmp_dir = output_path.parent / f".{output_path.stem}_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Count existing batches if resuming
    batch_num = 0
    if resume:
        existing_batches = sorted(tmp_dir.glob("batch_*.parquet"))
        batch_num = len(existing_batches)
        if batch_num > 0:
            print(f"Found {batch_num} existing batch files")

    # Batch buffer
    batch: list[tuple[bytes, int, int, int]] = []
    positions_extracted = progress.positions_extracted
    games_processed = progress.games_processed
    games_with_evals = 0

    def flush_batch() -> None:
        nonlocal batch, batch_num
        if not batch:
            return

        table = pyarrow.table(
            {
                "fen": [row[0] for row in batch],
                "score": [row[1] for row in batch],
                "depth": [row[2] for row in batch],
                "knodes": [row[3] for row in batch],
            },
            schema=arrow_schema,
        )

        batch_path = tmp_dir / f"batch_{batch_num:06d}.parquet"
        pyarrow.parquet.write_table(table, batch_path, compression="zstd")
        batch_num += 1
        batch = []

    def save_progress() -> None:
        progress.games_processed = games_processed
        progress.positions_extracted = positions_extracted
        progress.save(progress_path)

    def open_source() -> typing.IO[bytes]:
        """Open the source (URL or file) as a binary stream."""
        if is_url and use_torrent:
            print(f"Streaming via torrent: {source}.torrent")
            return download.open_torrent_stream(source)
        elif is_url:
            print(f"Streaming from {source}...")
            request = urllib.request.Request(source)
            # Use 300s timeout - large files (30+ GB) need longer for slow connections
            return urllib.request.urlopen(request, timeout=300)
        else:
            print(f"Processing {source}...")
            return open(source, "rb")

    if use_cpp_parser:
        print("Extracting positions with PGN evals (C++ parser, no engine needed)")
    else:
        print("Extracting positions with PGN evals (Python parser, no engine needed)")

    def should_process_game_from_headers(headers: dict[str, str]) -> bool:
        """Check if game should be processed based on parsed headers."""
        # Skip bot games
        if config.skip_bots:
            if headers.get("WhiteTitle") == "BOT" or headers.get("BlackTitle") == "BOT":
                return False

        # Check ratings
        try:
            white_elo = int(headers.get("WhiteElo", "0"))
            black_elo = int(headers.get("BlackElo", "0"))
        except ValueError:
            white_elo = black_elo = 0

        if config.min_elo is not None:
            if white_elo < config.min_elo or black_elo < config.min_elo:
                return False

        if config.max_elo is not None:
            if white_elo > config.max_elo or black_elo > config.max_elo:
                return False

        return True

    try:
        with open_source() as raw_stream:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(raw_stream) as reader:
                text_stream = io.TextIOWrapper(
                    reader, encoding="utf-8", errors="replace"
                )

                pbar = tqdm.auto.tqdm(
                    desc="Games",
                    unit=" games",
                    initial=progress.games_processed if resume else 0,
                    disable=disable_progress,
                )

                if use_cpp_parser:
                    # Fast path: use C++ parser
                    debug_pgn_path = pathlib.Path("/tmp/last_pgn.txt")
                    for pgn_text in iter_pgn_text(text_stream):
                        games_processed += 1
                        pbar.update(1)

                        # Print status every 20k games when progress bar disabled
                        if disable_progress and games_processed % 20000 == 0:
                            print(
                                f"Games: {games_processed}, positions: {positions_extracted}",
                                flush=True,
                            )

                        # Skip games if resuming
                        if resume and games_processed <= progress.games_processed:
                            continue

                        # Filter game by headers
                        headers = parse_pgn_headers(pgn_text)
                        if not should_process_game_from_headers(headers):
                            continue

                        # Write to /tmp before parsing so we can debug crashes
                        # Only in debug mode (disabled by default - slows down processing)
                        # debug_pgn_path.write_text(
                        #     f"Game #{games_processed}\n\n{pgn_text}"
                        # )

                        # Parse with C++ and get all positions with evals
                        positions = dummy_chess.parse_pgn_with_evals(pgn_text)
                        if not positions:
                            continue

                        # Filter by config constraints
                        valid_positions = []
                        for compressed_fen, score, ply_idx in positions:
                            if ply_idx < config.min_ply:
                                continue
                            # Note: can't check piece count without decompressing
                            # For speed, we skip that check in C++ path
                            valid_positions.append((compressed_fen, score, ply_idx))

                        if not valid_positions:
                            continue

                        # Pick one random position
                        compressed_fen, score, ply_idx = rng.choice(valid_positions)
                        games_with_evals += 1

                        # Add to batch (depth=0, knodes=0 for PGN eval)
                        batch.append((compressed_fen, score, 0, 0))
                        positions_extracted += 1

                        # Flush batch and save progress periodically
                        if len(batch) >= config.batch_size:
                            flush_batch()
                            save_progress()

                        # Check limit
                        if max_positions and positions_extracted >= max_positions:
                            break
                else:
                    # Slow path: use Python chess library
                    for game in iter_pgn_games(text_stream):
                        games_processed += 1
                        pbar.update(1)

                        # Skip games if resuming
                        if resume and games_processed <= progress.games_processed:
                            continue

                        # Filter game
                        if not should_process_game(game, config):
                            continue

                        # Extract position with PGN eval
                        result = extract_position_with_eval(game, config, rng)
                        if result is None:
                            continue

                        fen, score, ply_idx = result
                        games_with_evals += 1

                        # Compress FEN and add to batch
                        # Use depth=0 and knodes=0 to indicate PGN eval (not engine)
                        compressed_fen = dummy_chess.compress_fen(fen)
                        batch.append((compressed_fen, score, 0, 0))
                        positions_extracted += 1

                        # Flush batch and save progress periodically
                        if len(batch) >= config.batch_size:
                            flush_batch()
                            save_progress()

                        # Check limit
                        if max_positions and positions_extracted >= max_positions:
                            break

                pbar.close()
                print(f"\nFinished reading all games from stream.")

        # Final flush
        print(f"Flushing final batch ({len(batch)} positions)...")
        flush_batch()
        save_progress()
        print(f"Final flush complete.")

    except KeyboardInterrupt:
        print("\nInterrupted! Progress saved. Run with --resume to continue.")
        flush_batch()
        save_progress()
        raise

    except (urllib.error.URLError, OSError) as e:
        print(f"\nError: {e}")
        print("Progress saved. Run with --resume to continue.")
        flush_batch()
        save_progress()
        raise

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        if use_cpp_parser:
            print("Last PGN saved to /tmp/last_pgn.txt")
        flush_batch()
        save_progress()
        raise

    print(
        f"Processed {games_processed} games, {games_with_evals} had evals, "
        f"extracted {positions_extracted} positions"
    )

    # Combine batch files into final output
    batch_files = sorted(tmp_dir.glob("batch_*.parquet"))
    if not batch_files:
        print("No positions extracted")
        shutil.rmtree(tmp_dir)
        return 0

    print(f"Combining {len(batch_files)} batch files...")
    # Stream batches to output to avoid loading all into memory
    first_table = pyarrow.parquet.read_table(batch_files[0])
    with pyarrow.parquet.ParquetWriter(
        output_path, first_table.schema, compression="zstd"
    ) as writer:
        writer.write_table(first_table)
        del first_table
        for batch_file in batch_files[1:]:
            table = pyarrow.parquet.read_table(batch_file)
            writer.write_table(table)
            del table

    # Cleanup
    shutil.rmtree(tmp_dir)
    if progress_path.exists():
        progress_path.unlink()

    print(f"Saved {positions_extracted} positions -> {output_path}")
    return positions_extracted


def process_games(
    source: str,  # URL or file path
    output_path: pathlib.Path,
    engine_path: str,
    config: GameExtractionConfig,
    max_positions: int | None = None,
    seed: int = 42,
    resume: bool = False,
    use_torrent: bool = False,
) -> int:
    """
    Process PGN game archive, extracting random positions with engine evaluation.

    Supports both HTTP URLs (streaming), BitTorrent (streaming), and local files.
    Uses batch files for atomic writes and full resumability.

    Args:
        source: URL (http/https) or local file path to .pgn.zst
        output_path: Output parquet file
        engine_path: Path to Stockfish or other UCI engine
        config: Extraction configuration
        max_positions: Maximum positions to extract (None = all)
        seed: Random seed
        resume: Whether to resume from checkpoint
        use_torrent: Use BitTorrent for streaming (appends .torrent to URL)

    Returns:
        Number of positions extracted
    """
    rng = random.Random(seed)
    is_url = source.startswith("http://") or source.startswith("https://")

    # Progress tracking
    progress_path = output_path.with_suffix(".progress.json")
    progress = GameExtractionProgress(url_or_path=source)

    if resume and progress_path.exists():
        progress = GameExtractionProgress.load(progress_path)
        if progress.url_or_path != source:
            print("Warning: Progress file is for different source, starting fresh")
            progress = GameExtractionProgress(url_or_path=source)
        else:
            print(
                f"Resuming from game {progress.games_processed}, "
                f"position {progress.positions_extracted}"
            )

    # Start engine (single-threaded for best per-position performance)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": 1})
    print(f"Using UCI engine: {engine_path} (depth={config.depth})")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Parquet schema
    arrow_schema = pyarrow.schema(
        [
            ("fen", pyarrow.binary()),
            ("score", pyarrow.int64()),
            ("depth", pyarrow.int64()),
            ("knodes", pyarrow.int64()),
        ]
    )

    # Use temp directory for batch files (next to output, not /tmp)
    tmp_dir = output_path.parent / f".{output_path.stem}_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Count existing batches if resuming
    batch_num = 0
    if resume:
        existing_batches = sorted(tmp_dir.glob("batch_*.parquet"))
        batch_num = len(existing_batches)
        if batch_num > 0:
            print(f"Found {batch_num} existing batch files")

    # Batch buffer
    batch: list[tuple[bytes, int, int, int]] = []
    positions_extracted = progress.positions_extracted
    games_processed = progress.games_processed

    def flush_batch() -> None:
        nonlocal batch, batch_num
        if not batch:
            return

        table = pyarrow.table(
            {
                "fen": [row[0] for row in batch],
                "score": [row[1] for row in batch],
                "depth": [row[2] for row in batch],
                "knodes": [row[3] for row in batch],
            },
            schema=arrow_schema,
        )

        batch_path = tmp_dir / f"batch_{batch_num:06d}.parquet"
        pyarrow.parquet.write_table(table, batch_path, compression="zstd")
        batch_num += 1
        batch = []

    def save_progress() -> None:
        progress.games_processed = games_processed
        progress.positions_extracted = positions_extracted
        progress.save(progress_path)

    def open_source() -> typing.IO[bytes]:
        """Open the source (URL or file) as a binary stream."""
        if is_url and use_torrent:
            print(f"Streaming via torrent: {source}.torrent")
            return download.open_torrent_stream(source)
        elif is_url:
            print(f"Streaming from {source}...")
            request = urllib.request.Request(source)
            # Use 300s timeout - large files (30+ GB) need longer for slow connections
            return urllib.request.urlopen(request, timeout=300)
        else:
            print(f"Processing {source}...")
            return open(source, "rb")

    try:
        with open_source() as raw_stream:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(raw_stream) as reader:
                text_stream = io.TextIOWrapper(
                    reader, encoding="utf-8", errors="replace"
                )

                pbar = tqdm.auto.tqdm(
                    desc="Games",
                    unit=" games",
                    initial=progress.games_processed if resume else 0,
                )

                for game in iter_pgn_games(text_stream):
                    games_processed += 1
                    pbar.update(1)

                    # Skip games if resuming
                    if resume and games_processed <= progress.games_processed:
                        continue

                    # Filter game
                    if not should_process_game(game, config):
                        continue

                    # Extract position
                    result = extract_position_from_game(game, config, rng)
                    if result is None:
                        continue

                    fen, ply_idx = result

                    # Evaluate with engine
                    board = chess.Board(fen)
                    info = engine.analyse(board, chess.engine.Limit(depth=config.depth))
                    score_obj = info["score"].white()
                    if score_obj.is_mate():
                        mate_in = score_obj.mate()
                        score = 10000 if mate_in > 0 else -10000
                    else:
                        score = score_obj.score()
                        if score is None:
                            continue
                    score = max(-15000, min(15000, score))
                    depth_out = info.get("depth", config.depth)
                    knodes = info.get("nodes", 0) // 1000

                    # Compress FEN and add to batch
                    compressed_fen = dummy_chess.compress_fen(fen)
                    batch.append((compressed_fen, score, depth_out, knodes))
                    positions_extracted += 1

                    # Flush batch and save progress periodically
                    if len(batch) >= config.batch_size:
                        flush_batch()
                        save_progress()

                    # Check limit
                    if max_positions and positions_extracted >= max_positions:
                        break

                pbar.close()

        # Final flush
        flush_batch()
        save_progress()

    except KeyboardInterrupt:
        print("\nInterrupted! Progress saved. Run with --resume to continue.")
        flush_batch()
        save_progress()
        raise

    except (urllib.error.URLError, OSError) as e:
        print(f"\nError: {e}")
        print("Progress saved. Run with --resume to continue.")
        flush_batch()
        save_progress()
        raise

    finally:
        engine.quit()

    print(
        f"Processed {games_processed} games, extracted {positions_extracted} positions"
    )

    # Combine batch files into final output
    batch_files = sorted(tmp_dir.glob("batch_*.parquet"))
    if not batch_files:
        print("No positions extracted")
        shutil.rmtree(tmp_dir)
        return 0

    print(f"Combining {len(batch_files)} batch files...")
    # Stream batches to output to avoid loading all into memory
    first_table = pyarrow.parquet.read_table(batch_files[0])
    with pyarrow.parquet.ParquetWriter(
        output_path, first_table.schema, compression="zstd"
    ) as writer:
        writer.write_table(first_table)
        del first_table
        for batch_file in batch_files[1:]:
            table = pyarrow.parquet.read_table(batch_file)
            writer.write_table(table)
            del table

    # Cleanup
    shutil.rmtree(tmp_dir)
    if progress_path.exists():
        progress_path.unlink()

    print(f"Saved {positions_extracted} positions -> {output_path}")
    return positions_extracted


def download_month_games(
    month: str,
    output_dir: pathlib.Path,
    variant: str = "standard",
    use_torrent: bool = False,
) -> pathlib.Path | None:
    """
    Download a month's game archive from Lichess.

    Args:
        month: Month in YYYY-MM format
        output_dir: Directory to save the file
        variant: "standard" or "chess960"
        use_torrent: Use BitTorrent for download (recommended for large files)

    Returns:
        Path to downloaded file, or None if failed
    """
    return download.download_games_month(
        month=month,
        output_dir=output_dir,
        variant=variant,
        use_torrent=use_torrent,
    )


# =============================================================================
# Parquet shuffling (external sort)
# =============================================================================


def dedupe_parquet(
    input_path: str,
    output_path: str | None = None,
    num_buckets: int = 256,
    row_group_size: int = 100_000,
    resume: bool = False,
    max_bucket_memory_mb: int = 512,
) -> tuple[int, int, int]:
    """
    Deduplicate a parquet file based on the 'fen' column.

    Scales to 10B+ rows with constant memory using hierarchical bucketing:
    1. Compute optimal bucket count based on data size and memory limit
    2. Stream input, partition by hash into bucket files (IPC format for true append)
    3. For each bucket: load, dedupe, write to output
    4. If a bucket exceeds memory, recursively sub-partition it

    Args:
        input_path: Input parquet file
        output_path: Output parquet file (None = stats only, no output)
        num_buckets: Initial number of buckets (auto-adjusted based on data size)
        row_group_size: Rows per output row group
        resume: Resume from previous interrupted run
        max_bucket_memory_mb: Target max memory per bucket in MB (default 512)

    Returns:
        Tuple of (total_rows, unique_rows, duplicates_removed)

    Memory: O(max_bucket_memory_mb) regardless of input size
    Scales to: 10B+ rows
    """
    input_path = pathlib.Path(input_path)
    stats_only = output_path is None
    if not stats_only:
        output_path = pathlib.Path(output_path)

    # Create temp directory
    if output_path:
        tmp_dir = output_path.parent / f".{output_path.stem}_dedupe_tmp"
    else:
        tmp_dir = input_path.parent / f".{input_path.stem}_dedupe_tmp"

    # Progress tracking
    progress_file = tmp_dir / "_progress.json"
    phase1_complete = False
    processed_buckets: set[int] = set()

    if resume and tmp_dir.exists() and progress_file.exists():
        with open(progress_file) as f:
            progress_data = json.load(f)
        phase1_complete = progress_data.get("phase1_complete", False)
        processed_buckets = set(progress_data.get("processed_buckets", []))
        num_buckets = progress_data.get("num_buckets", num_buckets)
        if phase1_complete:
            print(
                f"Resuming Phase 2: {len(processed_buckets)}/{num_buckets} buckets done"
            )
        else:
            print("Resuming Phase 1...")
    elif not resume and tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)

    def save_progress(phase1_done: bool, buckets_done: set[int] | None = None):
        data = {"phase1_complete": phase1_done, "num_buckets": num_buckets}
        if buckets_done is not None:
            data["processed_buckets"] = list(buckets_done)
        with open(progress_file, "w") as f:
            json.dump(data, f)

    # Get input stats
    pf = pyarrow.parquet.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    num_row_groups = pf.metadata.num_row_groups

    # Estimate row size from first row group
    first_rg = pf.metadata.row_group(0)
    bytes_per_row = first_rg.total_byte_size / max(first_rg.num_rows, 1)
    # Add overhead for hash column (8 bytes)
    bytes_per_row = max(bytes_per_row, 40) + 8

    # Calculate optimal bucket count to keep each bucket under memory limit
    # IMPORTANT: Polars needs ~4x raw data size for unique() operation
    # (original + hash table + result + temporary buffers)
    memory_multiplier = 4
    max_rows_per_bucket = (max_bucket_memory_mb * 1024 * 1024) / (
        bytes_per_row * memory_multiplier
    )
    optimal_buckets = max(16, int(total_rows / max_rows_per_bucket) + 1)
    # Round up to power of 2 for better hash distribution
    optimal_buckets = 1 << (optimal_buckets - 1).bit_length()
    # Cap at reasonable maximum
    optimal_buckets = min(optimal_buckets, 8192)

    if not phase1_complete and optimal_buckets != num_buckets:
        print(
            f"Auto-adjusting buckets: {num_buckets} -> {optimal_buckets} "
            f"(targeting {max_bucket_memory_mb}MB/bucket with {memory_multiplier}x safety)"
        )
        num_buckets = optimal_buckets

    est_bucket_mb = total_rows * bytes_per_row / num_buckets / 1024 / 1024
    print(
        f"Input: {total_rows:,} rows, {num_row_groups} row groups, "
        f"~{bytes_per_row:.0f} bytes/row"
    )
    print(
        f"Using {num_buckets} buckets -> ~{total_rows / num_buckets:,.0f} rows/bucket "
        f"(~{est_bucket_mb:.0f}MB raw, ~{est_bucket_mb * memory_multiplier:.0f}MB working)"
    )

    # Phase 1: Partition into bucket files using IPC (true streaming append)
    if not phase1_complete:
        print(f"Phase 1: Partitioning into {num_buckets} buckets...")

        rows_written = 0
        base_schema: pyarrow.Schema | None = None

        # Open IPC writers (file handles only, minimal memory)
        bucket_writers: dict[int, pyarrow.ipc.RecordBatchFileWriter] = {}
        bucket_file_handles: dict[int, typing.IO] = {}

        def get_writer(bucket_idx: int, schema: pyarrow.Schema):
            if bucket_idx not in bucket_writers:
                bucket_file = tmp_dir / f"bucket_{bucket_idx:05d}.arrow"
                fh = open(bucket_file, "wb")
                bucket_file_handles[bucket_idx] = fh
                bucket_writers[bucket_idx] = pyarrow.ipc.new_file(fh, schema)
            return bucket_writers[bucket_idx]

        with tqdm.auto.tqdm(
            total=total_rows, desc="Partitioning", unit=" rows"
        ) as pbar:
            for rg_idx in range(num_row_groups):
                table = pf.read_row_group(rg_idx)
                df = polars.from_arrow(table)
                del table

                # Add hash and bucket
                df = df.with_columns(polars.col("fen").hash(seed=42).alias("_hash"))
                df = df.with_columns(
                    (polars.col("_hash") % num_buckets)
                    .cast(polars.UInt16)
                    .alias("_bucket")
                )

                if base_schema is None:
                    base_schema = df.drop("_bucket").to_arrow().schema

                # Partition and write
                partitions = df.partition_by("_bucket", as_dict=True)
                del df

                for bucket_key, bucket_df in partitions.items():
                    bucket_idx = (
                        int(bucket_key[0])
                        if isinstance(bucket_key, tuple)
                        else int(bucket_key)
                    )
                    bucket_df = bucket_df.drop("_bucket")
                    writer = get_writer(bucket_idx, base_schema)
                    writer.write_table(bucket_df.to_arrow())

                del partitions
                rg_rows = pf.metadata.row_group(rg_idx).num_rows
                rows_written += rg_rows
                pbar.update(rg_rows)

        # Close all writers
        for writer in bucket_writers.values():
            writer.close()
        for fh in bucket_file_handles.values():
            fh.close()
        bucket_writers.clear()
        bucket_file_handles.clear()

        phase1_complete = True
        save_progress(True, set())
        print(f"Phase 1 complete: {rows_written:,} rows partitioned")

    # Phase 2: Dedupe each bucket
    print("Phase 2: Deduplicating buckets...")

    bucket_files = sorted(tmp_dir.glob("bucket_*.arrow"))
    if not bucket_files:
        print("No data to dedupe")
        shutil.rmtree(tmp_dir)
        return 0, 0, 0

    unique_rows = 0

    # Setup output writer
    output_writer = None
    if not stats_only:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get schema from first bucket
        first_table = pyarrow.ipc.open_file(bucket_files[0]).read_all()
        first_df = polars.from_arrow(first_table)
        out_cols = [c for c in first_df.columns if c != "_hash"]
        arrow_schema = first_df.select(out_cols).to_arrow().schema
        del first_table, first_df
        output_writer = pyarrow.parquet.ParquetWriter(
            output_path, arrow_schema, compression="zstd"
        )

    with tqdm.auto.tqdm(
        total=len(bucket_files), desc="Deduping", unit=" buckets"
    ) as pbar:
        for bucket_file in bucket_files:
            bucket_idx = int(bucket_file.stem.split("_")[1])
            if bucket_idx in processed_buckets:
                pbar.update(1)
                continue

            # Read IPC file
            reader = pyarrow.ipc.open_file(bucket_file)
            table = reader.read_all()
            df = polars.from_arrow(table)
            del table, reader

            # Dedupe
            unique_df = df.unique(subset=["_hash"], keep="first")
            del df
            unique_rows += len(unique_df)

            # Write output
            if output_writer is not None:
                out_df = unique_df.drop("_hash")
                for i in range(0, len(out_df), row_group_size):
                    chunk = out_df.slice(i, row_group_size)
                    output_writer.write_table(chunk.to_arrow())
                del out_df

            del unique_df
            processed_buckets.add(bucket_idx)
            save_progress(True, processed_buckets)
            pbar.update(1)

    if output_writer is not None:
        output_writer.close()

    # Cleanup
    shutil.rmtree(tmp_dir)

    duplicates = total_rows - unique_rows
    print(f"Total: {total_rows:,}, Unique: {unique_rows:,}, Duplicates: {duplicates:,}")
    print(f"Duplicate rate: {100 * duplicates / total_rows:.2f}%")
    return total_rows, unique_rows, duplicates


def shuffle_parquet(
    input_path: str,
    output_path: str,
    num_buckets: int = 64,
    row_group_size: int = 100_000,
    seed: int | None = None,
    resume: bool = False,
    max_bucket_memory_mb: int = 512,
) -> int:
    """
    Shuffle a parquet file with minimal RAM using two-phase external sort.

    Scales to 10B+ rows with constant memory:
    1. Compute optimal bucket count based on data size and memory limit
    2. Stream input, add random sort key, partition by key range into IPC files
    3. For each bucket: load, sort by key, write to output

    Args:
        input_path: Input parquet file
        output_path: Output parquet file (will be overwritten)
        num_buckets: Initial number of buckets (auto-adjusted based on data size)
        row_group_size: Rows per output row group
        seed: Random seed for reproducibility
        resume: Resume from previous interrupted run
        max_bucket_memory_mb: Target max memory per bucket in MB (default 512)

    Returns:
        Number of rows written

    Memory: O(max_bucket_memory_mb) regardless of input size
    Scales to: 10B+ rows
    """
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)

    # Create temp directory
    tmp_dir = output_path.parent / f".{output_path.stem}_shuffle_tmp"

    # Progress tracking
    progress_file = tmp_dir / "_progress.json"
    phase1_complete = False
    processed_buckets: set[int] = set()

    if resume and tmp_dir.exists() and progress_file.exists():
        with open(progress_file) as f:
            progress_data = json.load(f)
        phase1_complete = progress_data.get("phase1_complete", False)
        processed_buckets = set(progress_data.get("processed_buckets", []))
        num_buckets = progress_data.get("num_buckets", num_buckets)
        seed = progress_data.get("seed", seed)
        if phase1_complete:
            print(
                f"Resuming Phase 2: {len(processed_buckets)}/{num_buckets} buckets done"
            )
        else:
            print("Resuming Phase 1...")
    elif not resume and tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)

    def save_progress(phase1_done: bool, buckets_done: set[int] | None = None):
        data = {
            "phase1_complete": phase1_done,
            "num_buckets": num_buckets,
            "seed": seed,
        }
        if buckets_done is not None:
            data["processed_buckets"] = list(buckets_done)
        with open(progress_file, "w") as f:
            json.dump(data, f)

    # Get input stats
    pf = pyarrow.parquet.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    num_row_groups = pf.metadata.num_row_groups

    # Estimate row size
    first_rg = pf.metadata.row_group(0)
    bytes_per_row = first_rg.total_byte_size / max(first_rg.num_rows, 1)
    bytes_per_row = max(bytes_per_row, 40) + 8  # +8 for sort key

    # Calculate optimal bucket count
    # Polars sort() needs ~3x raw data (original + sorted + temp)
    memory_multiplier = 3
    max_rows_per_bucket = (max_bucket_memory_mb * 1024 * 1024) / (
        bytes_per_row * memory_multiplier
    )
    optimal_buckets = max(16, int(total_rows / max_rows_per_bucket) + 1)
    optimal_buckets = 1 << (optimal_buckets - 1).bit_length()  # Round to power of 2
    optimal_buckets = min(optimal_buckets, 8192)

    if not phase1_complete and optimal_buckets != num_buckets:
        print(
            f"Auto-adjusting buckets: {num_buckets} -> {optimal_buckets} "
            f"(targeting {max_bucket_memory_mb}MB with {memory_multiplier}x safety)"
        )
        num_buckets = optimal_buckets

    est_bucket_mb = total_rows * bytes_per_row / num_buckets / 1024 / 1024
    print(f"Input: {total_rows:,} rows, {num_row_groups} row groups")
    print(
        f"Using {num_buckets} buckets -> ~{total_rows / num_buckets:,.0f} rows/bucket "
        f"(~{est_bucket_mb:.0f}MB raw, ~{est_bucket_mb * memory_multiplier:.0f}MB working)"
    )

    # Phase 1: Partition into bucket files with random sort keys
    if not phase1_complete:
        print(f"Phase 1: Partitioning into {num_buckets} buckets...")

        # Open IPC writers
        bucket_writers: dict[int, pyarrow.ipc.RecordBatchFileWriter] = {}
        bucket_file_handles: dict[int, typing.IO] = {}
        base_schema: pyarrow.Schema | None = None

        def get_writer(bucket_idx: int, schema: pyarrow.Schema):
            if bucket_idx not in bucket_writers:
                bucket_file = tmp_dir / f"bucket_{bucket_idx:05d}.arrow"
                fh = open(bucket_file, "wb")
                bucket_file_handles[bucket_idx] = fh
                bucket_writers[bucket_idx] = pyarrow.ipc.new_file(fh, schema)
            return bucket_writers[bucket_idx]

        with tqdm.auto.tqdm(
            total=total_rows, desc="Partitioning", unit=" rows"
        ) as pbar:
            for rg_idx in range(num_row_groups):
                table = pf.read_row_group(rg_idx)
                df = polars.from_arrow(table)
                del table

                # Add random sort key (deterministic based on seed + row index)
                n_rows = len(df)
                # Use hash of (seed, global_row_idx) for deterministic randomness
                row_offset = sum(
                    pf.metadata.row_group(i).num_rows for i in range(rg_idx)
                )
                sort_keys = [
                    hash((seed, row_offset + i)) % (2**63) for i in range(n_rows)
                ]
                df = df.with_columns(polars.Series("_sort_key", sort_keys))

                # Compute bucket from sort key
                df = df.with_columns(
                    (polars.col("_sort_key") % num_buckets)
                    .cast(polars.UInt16)
                    .alias("_bucket")
                )

                if base_schema is None:
                    base_schema = df.drop("_bucket").to_arrow().schema

                # Partition and write
                partitions = df.partition_by("_bucket", as_dict=True)
                del df

                for bucket_key, bucket_df in partitions.items():
                    bucket_idx = (
                        int(bucket_key[0])
                        if isinstance(bucket_key, tuple)
                        else int(bucket_key)
                    )
                    bucket_df = bucket_df.drop("_bucket")
                    writer = get_writer(bucket_idx, base_schema)
                    writer.write_table(bucket_df.to_arrow())

                del partitions
                pbar.update(pf.metadata.row_group(rg_idx).num_rows)

        # Close writers
        for writer in bucket_writers.values():
            writer.close()
        for fh in bucket_file_handles.values():
            fh.close()
        bucket_writers.clear()
        bucket_file_handles.clear()

        phase1_complete = True
        save_progress(True, set())
        print(f"Phase 1 complete")

    # Phase 2: Read buckets, sort, write to output
    print("Phase 2: Sorting and writing...")

    bucket_files = sorted(tmp_dir.glob("bucket_*.arrow"))
    if not bucket_files:
        print("No data to shuffle")
        shutil.rmtree(tmp_dir)
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get schema from first bucket (without _sort_key)
    first_reader = pyarrow.ipc.open_file(bucket_files[0])
    first_table = first_reader.read_all()
    first_df = polars.from_arrow(first_table)
    out_cols = [c for c in first_df.columns if c != "_sort_key"]
    arrow_schema = first_df.select(out_cols).to_arrow().schema
    del first_table, first_df, first_reader

    written = 0
    with pyarrow.parquet.ParquetWriter(
        output_path, arrow_schema, compression="zstd"
    ) as writer:
        with tqdm.auto.tqdm(
            total=len(bucket_files), desc="Writing", unit=" buckets"
        ) as pbar:
            for bucket_file in bucket_files:
                bucket_idx = int(bucket_file.stem.split("_")[1])
                if bucket_idx in processed_buckets:
                    pbar.update(1)
                    continue

                # Read and sort bucket
                reader = pyarrow.ipc.open_file(bucket_file)
                table = reader.read_all()
                df = polars.from_arrow(table).sort("_sort_key").drop("_sort_key")
                del table, reader

                # Write in chunks
                for i in range(0, len(df), row_group_size):
                    chunk = df.slice(i, row_group_size)
                    writer.write_table(chunk.to_arrow())
                written += len(df)

                del df
                processed_buckets.add(bucket_idx)
                save_progress(True, processed_buckets)
                pbar.update(1)

    # Cleanup
    shutil.rmtree(tmp_dir)

    print(f"Shuffled {written:,} rows -> {output_path}")
    return written


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
        choices=[
            "puzzles",
            "evals",
            "endgames",
            "games",
            "games-all",
            "shuffle",
            "dedupe",
        ],
        help="Data source: 'puzzles', 'evals', 'endgames', 'games', 'games-all', 'shuffle', or 'dedupe'",
    )
    parser.add_argument("-i", "--input", default=None, help="Input file")
    parser.add_argument("-o", "--output", default=None, help="Output file")
    parser.add_argument(
        "-n", "--max", type=int, default=None, help="Max rows to process/generate"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--toy", action="store_true", help="Generate toy dataset")

    # Puzzle/games-specific
    parser.add_argument("-e", "--engine", default=None, help="Path to UCI engine")
    parser.add_argument("-d", "--depth", type=int, default=8, help="UCI engine depth")

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

    # Games-specific
    parser.add_argument(
        "--month",
        default=None,
        help="Month to download/process (YYYY-MM format, e.g., 2024-01)",
    )
    parser.add_argument(
        "--after",
        default=None,
        metavar="YYYY-MM",
        help="Process months >= this date (for games-all)",
    )
    parser.add_argument(
        "--before",
        default=None,
        metavar="YYYY-MM",
        help="Process months < this date (for games-all)",
    )
    parser.add_argument(
        "--variant",
        default="standard",
        choices=["standard", "chess960"],
        help="Game variant (for games-all)",
    )
    parser.add_argument(
        "--list-months",
        action="store_true",
        help="List available months and exit (for games-all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess months even if output exists (for games-all)",
    )
    parser.add_argument(
        "--use-pgn-evals",
        action="store_true",
        help="Use [%%eval] annotations from PGN instead of engine (much faster)",
    )
    parser.add_argument(
        "--use-cpp-parser",
        action="store_true",
        help="Use fast C++ PGN parser (requires --use-pgn-evals)",
    )
    parser.add_argument(
        "--min-ply",
        type=int,
        default=10,
        help="Skip opening positions before this ply",
    )
    parser.add_argument(
        "--min-elo",
        type=int,
        default=None,
        help="Minimum player rating to include",
    )
    parser.add_argument(
        "--max-elo",
        type=int,
        default=None,
        help="Maximum player rating to include",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (for games)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download, don't process (for games)",
    )
    parser.add_argument(
        "--torrent",
        action="store_true",
        help="Use BitTorrent for download (recommended for large files)",
    )

    # Shuffle/dedupe-specific
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=256,
        help="Initial number of buckets for shuffle/dedupe (auto-adjusted for memory)",
    )
    parser.add_argument(
        "--row-group-size", type=int, default=100_000, help="Rows per output row group"
    )
    parser.add_argument(
        "--max-bucket-mb",
        type=int,
        default=512,
        help="Target max memory per bucket in MB (dedupe auto-adjusts bucket count)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="For dedupe: only count duplicates, don't write output",
    )
    parser.add_argument(
        "--resume-dedupe",
        action="store_true",
        help="Resume interrupted dedupe from checkpoint",
    )
    parser.add_argument(
        "--resume-shuffle",
        action="store_true",
        help="Resume interrupted shuffle from checkpoint",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
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
            engine_path=args.engine,
            engine_depth=args.depth,
        )

        print(f"Valid: {len(data)}")

        save_data(
            data,
            pathlib.Path(args.output),
            shuffle=not args.no_shuffle,
            seed=args.seed,
        )
        return

    elif args.source == "games":
        # Determine input source (URL or file path)
        input_source: str
        if args.input:
            input_source = args.input
            # Check if local file exists (skip check for URLs)
            if not (
                input_source.startswith("http://")
                or input_source.startswith("https://")
            ):
                input_path = pathlib.Path(input_source)
                if not input_path.exists():
                    print(f"Error: {input_path} not found")
                    sys.exit(1)
        elif args.month:
            # Download the month's games
            if args.month in BLOCKLISTED_MONTHS:
                print(f"Error: Month {args.month} is blocklisted due to data issues")
                print(f"Blocklisted months: {', '.join(sorted(BLOCKLISTED_MONTHS))}")
                sys.exit(1)

            download_dir = pathlib.Path("data/games")
            input_path = download_month_games(
                args.month, download_dir, use_torrent=args.torrent
            )
            if input_path is None:
                sys.exit(1)

            if args.download_only:
                print(f"Downloaded: {input_path}")
                return
            input_source = str(input_path)
        else:
            print("Error: --input or --month is required for games")
            print("Example: --month 2024-01")
            sys.exit(1)

        # Set output path
        if args.output is None:
            if args.month:
                args.output = f"data/games_{args.month.replace('-', '_')}.parquet"
            elif input_source.startswith("http"):
                # Extract filename from URL
                url_filename = input_source.split("/")[-1].replace(".pgn.zst", "")
                args.output = f"data/games_{url_filename}.parquet"
            else:
                args.output = f"data/games_{pathlib.Path(input_source).stem}.parquet"

        # Build config
        config = GameExtractionConfig(
            min_ply=args.min_ply,
            min_pieces=args.min_pieces if args.min_pieces else 7,
            min_elo=args.min_elo,
            max_elo=args.max_elo,
            depth=args.depth,
        )

        if args.use_pgn_evals:
            # Use evals from PGN annotations (no engine needed)
            process_games_pgn_evals(
                input_source,
                pathlib.Path(args.output),
                config,
                max_positions=args.max,
                seed=args.seed,
                resume=args.resume,
                use_torrent=args.torrent,
                use_cpp_parser=args.use_cpp_parser,
                disable_progress=args.no_progress,
            )
        else:
            # Use engine for evaluation
            if not args.engine:
                print("Error: --engine is required (or use --use-pgn-evals)")
                sys.exit(1)
            process_games(
                input_source,
                pathlib.Path(args.output),
                args.engine,
                config,
                max_positions=args.max,
                seed=args.seed,
                resume=args.resume,
                use_torrent=args.torrent,
            )
        return

    elif args.source == "games-all":
        # Get list of available months
        games_list = download.get_games_list(args.variant)
        if not games_list:
            print("Error: Failed to fetch games list from Lichess")
            sys.exit(1)

        # List mode (doesn't require engine)
        if args.list_months:
            blocklist = download.get_blocklist(args.variant)
            print(
                f"\nAvailable {args.variant} game archives ({len(games_list)} months):\n"
            )
            for month, url in games_list:
                status = " [BLOCKLISTED]" if month in blocklist else ""
                print(f"  {month}{status}")
            print(f"\nBlocklisted: {', '.join(sorted(blocklist)) or 'none'}")
            return

        # Engine required unless using PGN evals
        if not args.use_pgn_evals and not args.engine:
            print("Error: --engine is required (or use --use-pgn-evals)")
            sys.exit(1)

        # Filter months
        blocklist = download.get_blocklist(args.variant)
        months_to_process = []
        for month, url in games_list:
            if month in blocklist:
                continue
            if args.after and month < args.after:
                continue
            if args.before and month >= args.before:
                continue
            months_to_process.append((month, url))

        if not months_to_process:
            print("No months to process (check --after/--before filters)")
            sys.exit(0)

        # Sort oldest first for processing
        months_to_process.sort(key=lambda x: x[0])

        print(f"\nWill process {len(months_to_process)} month(s)")
        print(f"  Range: {months_to_process[0][0]} to {months_to_process[-1][0]}")
        if len(months_to_process) <= 10:
            print(f"  Months: {', '.join(m for m, _ in months_to_process)}")

        # Warn about low-eval months when using PGN evals
        if args.use_pgn_evals:
            low_eval_months = [m for m, _ in months_to_process if m < "2015-01"]
            if low_eval_months:
                print()
                print(
                    "WARNING: Early months (pre-2015) have very few [%eval] annotations."
                )
                print("         Consider using --after 2015-01 for better results.")
                print(f"         Affected: {', '.join(low_eval_months[:5])}", end="")
                if len(low_eval_months) > 5:
                    print(f" ... and {len(low_eval_months) - 5} more")
                else:
                    print()
        print()

        # Set output directory
        output_dir = (
            pathlib.Path(args.output) if args.output else pathlib.Path("data/games")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build config
        config = GameExtractionConfig(
            min_ply=args.min_ply,
            min_pieces=args.min_pieces if args.min_pieces else 7,
            min_elo=args.min_elo,
            max_elo=args.max_elo,
            depth=args.depth,
        )

        processed = []
        failed = []

        for month, url in months_to_process:
            output_path = output_dir / f"games_{month.replace('-', '_')}.parquet"

            # Skip if output exists (unless --force)
            if output_path.exists() and not args.force:
                print(
                    f"Skipping {month}: {output_path} exists (use --force to reprocess)"
                )
                processed.append(month)
                continue

            print(f"\n{'=' * 60}")
            print(f"Processing {month} ({args.variant})")
            print(f"  URL: {url}")
            if args.torrent:
                print(f"  Streaming via torrent")
            print(f"  Output: {output_path}")
            print()

            try:
                if args.use_pgn_evals:
                    process_games_pgn_evals(
                        source=url,
                        output_path=output_path,
                        config=config,
                        max_positions=args.max,
                        seed=args.seed,
                        resume=True,
                        use_torrent=args.torrent,
                        use_cpp_parser=args.use_cpp_parser,
                        disable_progress=args.no_progress,
                    )
                else:
                    process_games(
                        source=url,
                        output_path=output_path,
                        engine_path=args.engine,
                        config=config,
                        max_positions=args.max,
                        seed=args.seed,
                        resume=True,
                        use_torrent=args.torrent,
                    )
                processed.append(month)
            except KeyboardInterrupt:
                print(f"\nInterrupted during {month}. Progress saved.")
                raise
            except Exception as e:
                print(f"Error processing {month}: {e}")
                failed.append(month)

        print(f"\n{'=' * 60}")
        print(f"Processed: {len(processed)}, Failed: {len(failed)}")
        if failed:
            print(f"Failed months: {', '.join(failed)}")
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
            resume=args.resume_shuffle,
            max_bucket_memory_mb=args.max_bucket_mb,
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
            resume=args.resume_dedupe,
            max_bucket_memory_mb=args.max_bucket_mb,
        )
        return


if __name__ == "__main__":
    main()
