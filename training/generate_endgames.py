#!/usr/bin/env python3
"""
Tablebase-guided endgame position generator.

Generates random legal endgame positions (3-6 pieces) and labels them
with exact WDL values from Syzygy tablebases.
"""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.syzygy


# Default tablebase path relative to this file
DEFAULT_TABLEBASE_PATH = str(
    Path(__file__).parent.parent / "external" / "syzygy" / "src"
)

# Piece types excluding king (king is always present)
PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
PIECES_NO_PAWN = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


def detect_max_pieces(tablebase_path: str) -> int:
    """
    Detect the maximum piece count supported by available tablebases.

    Syzygy tablebase filenames encode the pieces, e.g.:
    - KQvK.rtbw = 3 pieces (KQ vs K)
    - KRPvKR.rtbw = 5 pieces (KRP vs KR)

    Returns the maximum piece count found (minimum 3).
    """
    tb_path = Path(tablebase_path)
    if not tb_path.exists():
        return 3

    max_pieces = 3

    # Look for .rtbw files (WDL tables)
    for rtbw_file in tb_path.glob("*.rtbw"):
        name = rtbw_file.stem  # e.g., "KRPvKR"

        # Count pieces: each letter except 'v' is a piece
        # K=King, Q=Queen, R=Rook, B=Bishop, N=Knight, P=Pawn
        piece_count = sum(1 for c in name if c in "KQRBNPkqrbnp")
        max_pieces = max(max_pieces, piece_count)

    return max_pieces


def get_tablebase_info(tablebase_path: str | None = None) -> dict:
    """
    Get information about available tablebases.

    Returns:
        Dict with 'max_pieces', 'num_tables', and 'path'
    """
    if tablebase_path is None:
        tablebase_path = DEFAULT_TABLEBASE_PATH

    tb_path = Path(tablebase_path)
    rtbw_files = list(tb_path.glob("*.rtbw")) if tb_path.exists() else []

    return {
        "path": tablebase_path,
        "num_tables": len(rtbw_files),
        "max_pieces": detect_max_pieces(tablebase_path),
    }


@dataclass
class EndgameConfig:
    """Configuration for endgame generation."""

    min_pieces: int = 3  # Minimum total pieces (including kings)
    max_pieces: int | None = None  # None = auto-detect from tablebases
    include_pawns: bool = True  # Whether to include pawn endgames

    def resolve_max_pieces(self, tablebase_path: str) -> int:
        """Get max_pieces, auto-detecting from tablebases if None."""
        if self.max_pieces is not None:
            return self.max_pieces
        return detect_max_pieces(tablebase_path)


def random_square(
    rng: random.Random, exclude: set[int] | None = None, pawn: bool = False
) -> int:
    """Generate a random square, optionally excluding some and handling pawn restrictions."""
    if exclude is None:
        exclude = set()

    if pawn:
        # Pawns can't be on 1st or 8th rank
        valid = [sq for sq in range(8, 56) if sq not in exclude]
    else:
        valid = [sq for sq in range(64) if sq not in exclude]

    return rng.choice(valid) if valid else -1


def generate_random_position(
    config: EndgameConfig, max_pieces: int, rng: random.Random | None = None
) -> chess.Board | None:
    """
    Generate a random endgame position.

    Args:
        config: Generation configuration
        max_pieces: Maximum pieces (resolved, not None)
        rng: Random number generator (uses new unseeded Random if None)

    Returns None if generation fails (e.g., can't place pieces legally).
    """
    if rng is None:
        rng = random.Random()

    # Decide number of pieces (including 2 kings)
    n_pieces = rng.randint(config.min_pieces, max_pieces)
    n_extra = n_pieces - 2  # Extra pieces beyond the two kings

    # Create empty board
    board = chess.Board.empty()

    # Place kings
    occupied: set[int] = set()
    wk_sq = random_square(rng, occupied)
    occupied.add(wk_sq)

    # Black king must not be adjacent to white king
    adjacent = set()
    for d in [-9, -8, -7, -1, 1, 7, 8, 9]:
        adj = wk_sq + d
        if 0 <= adj < 64:
            # Check we don't wrap around the board
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

    # Distribute extra pieces between white and black
    pieces_to_place: list[tuple[int, bool]] = []  # (piece_type, is_white)

    for _ in range(n_extra):
        color = rng.choice([True, False])
        if config.include_pawns:
            piece_type = rng.choice(PIECES)
        else:
            piece_type = rng.choice(PIECES_NO_PAWN)
        pieces_to_place.append((piece_type, color))

    # Place the pieces
    for piece_type, is_white in pieces_to_place:
        is_pawn = piece_type == chess.PAWN
        sq = random_square(rng, occupied, pawn=is_pawn)
        if sq == -1:
            return None
        occupied.add(sq)
        board.set_piece_at(sq, chess.Piece(piece_type, is_white))

    # Set side to move randomly
    board.turn = rng.choice([chess.WHITE, chess.BLACK])

    # Clear castling rights (endgame)
    board.castling_rights = chess.BB_EMPTY

    # Validate the position
    if not board.is_valid():
        return None

    # Check the position isn't already in check for the non-moving side
    # (which would be illegal)
    board.turn = not board.turn
    if board.is_check():
        return None
    board.turn = not board.turn

    return board


def wdl_to_score(wdl: int, turn: bool) -> float:
    """
    Convert tablebase WDL to a score.

    WDL values from python-chess:
      2 = win, 1 = cursed win, 0 = draw, -1 = blessed loss, -2 = loss

    Returns score from white's perspective in centipawns.
    """
    # Map WDL to approximate centipawn values
    wdl_scores = {
        2: 10000.0,  # Win
        1: 5000.0,  # Cursed win (win but 50-move rule)
        0: 0.0,  # Draw
        -1: -5000.0,  # Blessed loss
        -2: -10000.0,  # Loss
    }
    score = wdl_scores.get(wdl, 0.0)

    # Score is from side-to-move perspective, convert to white's perspective
    if not turn:  # Black to move
        score = -score

    return score


def generate_endgames(
    n_positions: int,
    tablebase_path: str | None = None,
    config: EndgameConfig | None = None,
    max_attempts_per_position: int = 100,
) -> list[tuple[str, float, int]]:
    """
    Generate endgame positions with tablebase evaluations.

    Args:
        n_positions: Number of positions to generate
        tablebase_path: Path to Syzygy tablebase files
        config: Generation configuration
        max_attempts_per_position: Max attempts before giving up on a position

    Returns:
        List of (fen, score, wdl) tuples
    """
    if config is None:
        config = EndgameConfig()

    if tablebase_path is None:
        tablebase_path = DEFAULT_TABLEBASE_PATH

    # Resolve max_pieces from tablebases if not specified
    max_pieces = config.resolve_max_pieces(tablebase_path)

    # Open tablebase
    tablebase = chess.syzygy.Tablebase()
    tablebase.add_directory(tablebase_path)

    results: list[tuple[str, float, int]] = []
    attempts = 0
    max_total_attempts = n_positions * max_attempts_per_position * 10

    while len(results) < n_positions and attempts < max_total_attempts:
        attempts += 1

        board = generate_random_position(config, max_pieces)
        if board is None:
            continue

        # Probe tablebase
        try:
            wdl = tablebase.probe_wdl(board)
            if wdl is None:
                continue
        except chess.syzygy.MissingTableError:
            continue
        except Exception:
            continue

        score = wdl_to_score(wdl, board.turn)
        fen = board.fen()
        results.append((fen, score, wdl))

    tablebase.close()
    return results


class EndgameDataset:
    """
    Iterable dataset that generates endgame positions on-the-fly.

    Can be used as a data source for training. Supports deterministic
    generation via seed parameter.

    Args:
        tablebase_path: Path to Syzygy tablebases (auto-detected if None)
        config: Generation configuration
        positions_per_epoch: Number of positions per iteration
        seed: Random seed for deterministic generation (None for non-deterministic)
    """

    def __init__(
        self,
        tablebase_path: str | None = None,
        config: EndgameConfig | None = None,
        positions_per_epoch: int = 100000,
        seed: int | None = None,
    ):
        self.tablebase_path = tablebase_path or DEFAULT_TABLEBASE_PATH
        self.config = config or EndgameConfig()
        self.positions_per_epoch = positions_per_epoch
        self.seed = seed
        self._tablebase: chess.syzygy.Tablebase | None = None
        self._max_pieces: int | None = None

    def _get_tablebase(self) -> chess.syzygy.Tablebase:
        if self._tablebase is None:
            self._tablebase = chess.syzygy.Tablebase()
            self._tablebase.add_directory(self.tablebase_path)
        return self._tablebase

    def _get_max_pieces(self) -> int:
        if self._max_pieces is None:
            self._max_pieces = self.config.resolve_max_pieces(self.tablebase_path)
        return self._max_pieces

    def __iter__(self):
        tb = self._get_tablebase()
        max_pieces = self._get_max_pieces()
        rng = random.Random(self.seed)

        count = 0
        max_attempts = self.positions_per_epoch * 100

        attempts = 0
        while count < self.positions_per_epoch and attempts < max_attempts:
            attempts += 1

            board = generate_random_position(self.config, max_pieces, rng)
            if board is None:
                continue

            try:
                wdl = tb.probe_wdl(board)
                if wdl is None:
                    continue
            except Exception:
                continue

            score = wdl_to_score(wdl, board.turn)
            yield board.fen(), score, wdl
            count += 1

    def __len__(self):
        return self.positions_per_epoch


class PuzzleDataset:
    """
    Iterable dataset that reads FEN/score pairs from a CSV file.

    Compatible with Lichess puzzle format (expects 'fen' and 'score' columns).

    Args:
        path: Path to CSV file (supports .gz compression)
        positions_per_epoch: Number of positions per iteration (None = all)
        seed: Random seed for shuffling (None for sequential, no shuffle)
    """

    def __init__(
        self,
        path: str,
        positions_per_epoch: int | None = None,
        seed: int | None = None,
    ):
        self.path = path
        self.positions_per_epoch = positions_per_epoch
        self.seed = seed

    def __iter__(self):
        import pandas as pd

        compression = "gzip" if self.path.endswith(".gz") else None
        df = pd.read_csv(self.path, compression=compression)

        if self.seed is not None:
            df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        count = 0
        limit = self.positions_per_epoch or len(df)

        for _, row in df.iterrows():
            if count >= limit:
                break
            try:
                fen = row["fen"]
                score = float(row["score"])
                yield fen, score
                count += 1
            except Exception:
                continue

    def __len__(self):
        if self.positions_per_epoch is not None:
            return self.positions_per_epoch
        import polars as pl

        return pl.scan_csv(self.path).select(pl.len()).collect().item()


class MixedDataset:
    """
    Dataset that mixes positions from multiple sources with deterministic interleaving.

    Combines puzzle data (CSV) and generated endgames with configurable ratios.

    Args:
        puzzle_path: Path to puzzle CSV file (expects 'fen' and 'score' columns)
        endgame_ratio: Fraction of positions from endgame generator (0.0-1.0)
        positions_per_epoch: Total positions per iteration
        seed: Random seed for deterministic mixing and generation
        tablebase_path: Path to Syzygy tablebases for endgame generation
        endgame_config: Configuration for endgame generation

    Example:
        dataset = MixedDataset(
            puzzle_path="data/puzzles.csv.gz",
            endgame_ratio=0.3,  # 30% endgames, 70% puzzles
            positions_per_epoch=100000,
            seed=42,
        )
        for fen, score in dataset:
            ...
    """

    def __init__(
        self,
        puzzle_path: str,
        endgame_ratio: float = 0.3,
        positions_per_epoch: int = 100000,
        seed: int | None = None,
        tablebase_path: str | None = None,
        endgame_config: EndgameConfig | None = None,
    ):
        if not 0.0 <= endgame_ratio <= 1.0:
            raise ValueError("endgame_ratio must be between 0.0 and 1.0")

        self.puzzle_path = puzzle_path
        self.endgame_ratio = endgame_ratio
        self.positions_per_epoch = positions_per_epoch
        self.seed = seed
        self.tablebase_path = tablebase_path or DEFAULT_TABLEBASE_PATH
        self.endgame_config = endgame_config or EndgameConfig()

        self._tablebase: chess.syzygy.Tablebase | None = None
        self._max_pieces: int | None = None

    def _get_tablebase(self) -> chess.syzygy.Tablebase:
        if self._tablebase is None:
            self._tablebase = chess.syzygy.Tablebase()
            self._tablebase.add_directory(self.tablebase_path)
        return self._tablebase

    def _get_max_pieces(self) -> int:
        if self._max_pieces is None:
            self._max_pieces = self.endgame_config.resolve_max_pieces(
                self.tablebase_path
            )
        return self._max_pieces

    def __iter__(self):
        import pandas as pd

        rng = random.Random(self.seed)

        # Load puzzle data
        compression = "gzip" if self.puzzle_path.endswith(".gz") else None
        puzzle_df = pd.read_csv(self.puzzle_path, compression=compression)

        # Shuffle puzzles deterministically
        if self.seed is not None:
            puzzle_df = puzzle_df.sample(frac=1, random_state=self.seed).reset_index(
                drop=True
            )

        puzzle_iter = puzzle_df.iterrows()
        puzzle_exhausted = False

        # Setup endgame generation
        tb = self._get_tablebase()
        max_pieces = self._get_max_pieces()

        count = 0
        max_attempts_per_endgame = 100

        while count < self.positions_per_epoch:
            # Decide source based on ratio
            use_endgame = rng.random() < self.endgame_ratio

            if use_endgame:
                # Generate endgame position
                for _ in range(max_attempts_per_endgame):
                    board = generate_random_position(
                        self.endgame_config, max_pieces, rng
                    )
                    if board is None:
                        continue
                    try:
                        wdl = tb.probe_wdl(board)
                        if wdl is None:
                            continue
                        score = wdl_to_score(wdl, board.turn)
                        yield board.fen(), score
                        count += 1
                        break
                    except Exception:
                        continue
            else:
                # Get puzzle position
                if puzzle_exhausted:
                    # Restart puzzle iteration
                    if self.seed is not None:
                        puzzle_df = puzzle_df.sample(
                            frac=1, random_state=self.seed + count
                        ).reset_index(drop=True)
                    puzzle_iter = puzzle_df.iterrows()
                    puzzle_exhausted = False

                try:
                    _, row = next(puzzle_iter)
                    fen = row["fen"]
                    score = float(row["score"])
                    yield fen, score
                    count += 1
                except StopIteration:
                    puzzle_exhausted = True
                    continue
                except Exception:
                    continue

    def __len__(self):
        return self.positions_per_epoch


class MixedTrainingDataset:
    """
    PyTorch-compatible dataset that yields HalfKP features for training.

    Combines puzzle data and generated endgames, yielding (w_feats, b_feats, stm, score)
    tuples compatible with train.py's collate_sparse function.

    Args:
        puzzle_path: Path to puzzle CSV file (expects 'fen' and 'score' columns)
        endgame_ratio: Fraction of positions from endgame generator (0.0-1.0)
        positions_per_epoch: Total positions per iteration
        seed: Random seed for deterministic mixing and generation
        tablebase_path: Path to Syzygy tablebases for endgame generation
        endgame_config: Configuration for endgame generation

    Example:
        from generate_endgames import MixedTrainingDataset
        from train import collate_sparse

        dataset = MixedTrainingDataset(
            puzzle_path="data/puzzles.csv.gz",
            endgame_ratio=0.3,
            positions_per_epoch=100000,
            seed=42,
        )
        loader = DataLoader(dataset, batch_size=8192, collate_fn=collate_sparse)
    """

    # HalfKP feature extraction constants (must match train.py)
    PIECE_TO_INDEX = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
        [-1, -1],  # P, N, B, R, Q, K
    ]

    def __init__(
        self,
        puzzle_path: str,
        endgame_ratio: float = 0.3,
        positions_per_epoch: int = 100000,
        seed: int | None = None,
        tablebase_path: str | None = None,
        endgame_config: EndgameConfig | None = None,
    ):
        if not 0.0 <= endgame_ratio <= 1.0:
            raise ValueError("endgame_ratio must be between 0.0 and 1.0")

        self.puzzle_path = puzzle_path
        self.endgame_ratio = endgame_ratio
        self.positions_per_epoch = positions_per_epoch
        self.seed = seed
        self.tablebase_path = tablebase_path or DEFAULT_TABLEBASE_PATH
        self.endgame_config = endgame_config or EndgameConfig()

        self._tablebase: chess.syzygy.Tablebase | None = None
        self._max_pieces: int | None = None

    def _get_tablebase(self) -> chess.syzygy.Tablebase:
        if self._tablebase is None:
            self._tablebase = chess.syzygy.Tablebase()
            self._tablebase.add_directory(self.tablebase_path)
        return self._tablebase

    def _get_max_pieces(self) -> int:
        if self._max_pieces is None:
            self._max_pieces = self.endgame_config.resolve_max_pieces(
                self.tablebase_path
            )
        return self._max_pieces

    @staticmethod
    def get_halfkp_features(fen: str) -> tuple[list[int], list[int], int]:
        """Extract HalfKP features from FEN (same as train.py)."""
        board = chess.Board(fen)
        wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)

        white_feats, black_feats = [], []
        for sq in chess.SQUARES:
            pc = board.piece_at(sq)
            if pc is None or pc.piece_type == chess.KING:
                continue
            pt = pc.piece_type - 1
            is_white = pc.color

            white_feats.append(
                wk * 641
                + MixedTrainingDataset.PIECE_TO_INDEX[pt][0 if is_white else 1] * 64
                + sq
                + 1
            )
            black_feats.append(
                (63 - bk) * 641
                + MixedTrainingDataset.PIECE_TO_INDEX[pt][1 if is_white else 0] * 64
                + (63 - sq)
                + 1
            )

        return white_feats, black_feats, 0 if board.turn else 1

    def __iter__(self):
        import pandas as pd

        rng = random.Random(self.seed)

        # Load puzzle data
        compression = "gzip" if self.puzzle_path.endswith(".gz") else None
        puzzle_df = pd.read_csv(self.puzzle_path, compression=compression)

        # Shuffle puzzles deterministically
        if self.seed is not None:
            puzzle_df = puzzle_df.sample(frac=1, random_state=self.seed).reset_index(
                drop=True
            )

        puzzle_iter = puzzle_df.iterrows()
        puzzle_exhausted = False

        # Setup endgame generation
        tb = self._get_tablebase()
        max_pieces = self._get_max_pieces()

        count = 0
        max_attempts_per_endgame = 100

        while count < self.positions_per_epoch:
            # Decide source based on ratio
            use_endgame = rng.random() < self.endgame_ratio

            if use_endgame:
                # Generate endgame position
                for _ in range(max_attempts_per_endgame):
                    board = generate_random_position(
                        self.endgame_config, max_pieces, rng
                    )
                    if board is None:
                        continue
                    try:
                        wdl = tb.probe_wdl(board)
                        if wdl is None:
                            continue
                        score = wdl_to_score(wdl, board.turn)
                        w, b, stm = self.get_halfkp_features(board.fen())
                        yield w, b, stm, score
                        count += 1
                        break
                    except Exception:
                        continue
            else:
                # Get puzzle position
                if puzzle_exhausted:
                    # Restart puzzle iteration with new shuffle
                    if self.seed is not None:
                        puzzle_df = puzzle_df.sample(
                            frac=1, random_state=self.seed + count
                        ).reset_index(drop=True)
                    puzzle_iter = puzzle_df.iterrows()
                    puzzle_exhausted = False

                try:
                    _, row = next(puzzle_iter)
                    fen = row["fen"]
                    score = float(row["score"])
                    w, b, stm = self.get_halfkp_features(fen)
                    yield w, b, stm, score
                    count += 1
                except StopIteration:
                    puzzle_exhausted = True
                    continue
                except Exception:
                    continue

    def __len__(self):
        return self.positions_per_epoch


def main():
    parser = argparse.ArgumentParser(description="Generate tablebase-labeled endgames")
    parser.add_argument(
        "--tablebase",
        "-t",
        default=DEFAULT_TABLEBASE_PATH,
        help=f"Path to Syzygy tablebase directory (default: {DEFAULT_TABLEBASE_PATH})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="endgames.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=10000,
        help="Number of positions to generate",
    )
    parser.add_argument(
        "--min-pieces",
        type=int,
        default=3,
        help="Minimum pieces (including kings)",
    )
    parser.add_argument(
        "--max-pieces",
        type=int,
        default=None,
        help="Maximum pieces (including kings, auto-detected from tablebases if not specified)",
    )
    parser.add_argument(
        "--no-pawns",
        action="store_true",
        help="Exclude pawn endgames",
    )
    args = parser.parse_args()

    config = EndgameConfig(
        min_pieces=args.min_pieces,
        max_pieces=args.max_pieces,
        include_pawns=not args.no_pawns,
    )

    # Get tablebase info
    tb_info = get_tablebase_info(args.tablebase)
    max_pieces = config.resolve_max_pieces(args.tablebase)

    print(f"Generating {args.count} endgame positions...")
    print(f"  Tablebase: {args.tablebase}")
    print(f"  Tables found: {tb_info['num_tables']}")
    print(
        f"  Pieces: {config.min_pieces}-{max_pieces}"
        + (" (auto-detected)" if args.max_pieces is None else "")
    )
    print(f"  Pawns: {'yes' if config.include_pawns else 'no'}")

    positions = generate_endgames(args.count, args.tablebase, config)

    print(f"Generated {len(positions)} positions")

    # Write to CSV
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        f.write("fen,score,wdl\n")
        for fen, score, wdl in positions:
            f.write(f"{fen},{score},{wdl}\n")

    print(f"Saved to {output_path}")

    # Print WDL distribution
    wdl_counts = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}
    for _, _, wdl in positions:
        wdl_counts[wdl] += 1
    print("\nWDL distribution:")
    print(f"  Wins:    {wdl_counts[2]}")
    print(f"  Cursed:  {wdl_counts[1]}")
    print(f"  Draws:   {wdl_counts[0]}")
    print(f"  Blessed: {wdl_counts[-1]}")
    print(f"  Losses:  {wdl_counts[-2]}")


if __name__ == "__main__":
    main()
