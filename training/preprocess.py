#!/usr/bin/env python3
"""
Preprocess data for NNUE training.

Supports three data sources:
1. puzzles: lichess_db_puzzle.csv.zst - uses Stockfish or material eval
2. evals: lichess_db_eval.jsonl.zst - uses pre-computed Stockfish evals
3. endgames: generates random endgame positions using Syzygy tablebases

Outputs a single CSV file with columns: fen, score
"""

import argparse
import io
import itertools
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chess
import chess.engine
import chess.syzygy
import dummy_chess
import orjson
import polars as pl
import zstandard as zstd
from tqdm.auto import tqdm


# =============================================================================
# LazyNdjsonZstd - streaming zstd NDJSON reader
# =============================================================================


class LazyNdjsonZstd:
    """
    LazyFrame-like interface for zstd-compressed NDJSON files.

    Uses streaming decompression to stay within memory limits.
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
        rows = list(itertools.islice(self._iter_lines(), n))
        return pl.from_dicts(rows)

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
            return pl.from_dicts(rows)

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
    """Scan a zstd-compressed NDJSON file."""
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


def stockfish_eval(
    board: chess.Board,
    engine,
    depth: int = 12,
) -> tuple[int, int, int]:
    """Stockfish evaluation. Returns (score_cp, depth, knodes)."""
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
    row: dict, engine=None, depth: int = 12
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
    if engine:
        score, out_depth, out_knodes = stockfish_eval(board, engine, depth)
    else:
        score = material_eval(board)
        out_depth = 0
        out_knodes = 0
    score = max(-15000, min(15000, score))
    results.append((board.fen(), score, out_depth, out_knodes))

    # Position 2: After move 1 (opponent's mistake)
    move = chess.Move.from_uci(moves[0])
    if move not in board.legal_moves:
        return results  # Return just the starting position
    board.push(move)

    if engine:
        score, out_depth, out_knodes = stockfish_eval(board, engine, depth)
    else:
        score = material_eval(board)
        out_depth = 0
        out_knodes = 0
    score = max(-15000, min(15000, score))
    results.append((board.fen(), score, out_depth, out_knodes))

    return results


def process_puzzles(
    input_path: Path,
    output_path: Path,
    max_rows: int | None,
    stockfish_path: str | None,
    depth: int,
    batch_size: int = 100000,
) -> int:
    """
    Process puzzle data, streaming to parquet in batches.

    Returns the number of rows written.
    """
    import shutil
    import tempfile

    engine = None
    if stockfish_path:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            print(f"Using Stockfish: {stockfish_path} (depth={depth})")
        except Exception as e:
            print(f"Stockfish failed ({e}), using material eval")

    print(f"Loading {input_path}...")
    if str(input_path).endswith(".zst"):
        lf = pl_scan_csv_zstd(input_path)
    else:
        lf = pl.scan_csv(input_path)

    if max_rows:
        lf = lf.head(max_rows)

    # Pre-count total rows for progress bar (streaming to avoid memory issues)
    total_rows = lf.select(pl.len()).collect(engine="streaming").item()
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
        df = pl.DataFrame(
            compressed_batch,
            schema={
                "fen": pl.Binary,
                "score": pl.Int64,
                "depth": pl.Int64,
                "knodes": pl.Int64,
            },
            orient="row",
        )
        df.write_parquet(f"{tmp_dir}/batch_{batch_num:06d}.parquet")
        batch_num += 1
        batch = []

    pbar = tqdm(total=total_rows, desc="Puzzles", unit=" rows")
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
    lf_out = pl.scan_parquet(f"{tmp_dir}/*.parquet")

    out_str = str(output_path)
    if out_str.endswith(".parquet") or not any(
        out_str.endswith(ext) for ext in [".csv", ".csv.gz", ".csv.zst"]
    ):
        lf_out.sink_parquet(output_path)
    else:
        # CSV requires collect
        combined = lf_out.collect()
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
    try:
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
    except Exception:
        return None


def process_evals(
    input_path: Path,
    output_path: Path,
    max_rows: int | None,
    shuffle: bool = True,
    seed: int = 42,
    batch_size: int = 1000000,
) -> int:
    """
    Process evaluation data, streaming to parquet in batches.

    Returns the number of rows written.
    """
    import shutil
    import subprocess
    import tempfile

    print(f"Loading {input_path}...")
    lf = pl_scan_ndjson_zstd(input_path)

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
        df = pl.DataFrame(
            compressed_batch,
            schema={
                "fen": pl.Binary,
                "score": pl.Int64,
                "depth": pl.Int64,
                "knodes": pl.Int64,
            },
            orient="row",
        )
        df.write_parquet(f"{tmp_dir}/batch_{batch_num:06d}.parquet")
        batch_num += 1
        batch = []

    pbar = tqdm(desc="Evals", unit=" rows")
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
    lf = pl.scan_parquet(f"{tmp_dir}/*.parquet")

    out_str = str(output_path)
    if out_str.endswith(".parquet") or not any(
        out_str.endswith(ext) for ext in [".csv", ".csv.gz", ".csv.zst"]
    ):
        # Stream directly to parquet without loading into memory
        lf.sink_parquet(output_path)
    else:
        # CSV requires collect (no sink_csv for LazyFrame with glob)
        combined = lf.collect()
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
    Path(__file__).parent.parent / "external" / "syzygy" / "src"
)

PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
PIECES_NO_PAWN = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


@dataclass
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
    tb_path = Path(tablebase_path)
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

    pbar = tqdm(total=n_positions, desc="Endgames")

    try:
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
            except Exception:
                continue

            fen_results.append((board.fen(), int(score), TABLEBASE_DEPTH, 0))
            pbar.update(1)
    finally:
        pbar.close()
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
# Output
# =============================================================================


def save_data(
    data: list[tuple[bytes, int, int, int]],
    output_path: Path,
    shuffle: bool = True,
    seed: int = 42,
):
    """Save data to parquet or CSV file with columns: fen (binary), score, depth, knodes."""
    if shuffle:
        random.seed(seed)
        random.shuffle(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_df = pl.DataFrame(
        data,
        schema={
            "fen": pl.Binary,
            "score": pl.Int64,
            "depth": pl.Int64,
            "knodes": pl.Int64,
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
        choices=["puzzles", "evals", "endgames"],
        help="Data source: 'puzzles', 'evals', or 'endgames'",
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
    parser.add_argument("-s", "--stockfish", default=None, help="Path to stockfish")
    parser.add_argument("-d", "--depth", type=int, default=12, help="Stockfish depth")

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
        output_path = Path(args.output)
        args.output = str(output_path.parent / f"toy_{output_path.name}")

    # Process
    data: list[tuple[bytes, int, int, int]] = []

    if args.source == "puzzles":
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: {args.input} not found")
            print(
                "Download: wget https://database.lichess.org/lichess_db_puzzle.csv.zst"
            )
            sys.exit(1)
        # Puzzles handles its own output (streaming)
        process_puzzles(
            input_path,
            Path(args.output),
            args.max,
            args.stockfish,
            args.depth,
        )
        return

    elif args.source == "evals":
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: {args.input} not found")
            print(
                "Download: wget https://database.lichess.org/lichess_db_eval.jsonl.zst"
            )
            sys.exit(1)
        # Evals handles its own output (streaming)
        process_evals(
            input_path,
            Path(args.output),
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
        Path(args.output),
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
