#!/usr/bin/env python3
"""
Preprocess Lichess puzzles for NNUE training.

Generates train/val/test splits from puzzle CSV.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import chess
import pandas as pd
from tqdm import tqdm

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


def stockfish_eval(board: chess.Board, engine) -> int:
    """Stockfish evaluation in centipawns."""
    info = engine.analyse(board, chess.engine.Limit(depth=12))
    score = info["score"].white()
    if score.is_mate():
        return 10000 if score.mate() > 0 else -10000
    return score.score()


def process_puzzle(line: str, engine=None):
    """Process single puzzle line. Returns (fen, score) or None."""
    try:
        parts = line.strip().split(",")
        if len(parts) < 3:
            return None

        fen, moves = parts[1], parts[2].split()
        if not fen or not moves:
            return None

        board = chess.Board(fen)
        for m in moves:
            move = chess.Move.from_uci(m)
            if move not in board.legal_moves:
                return None
            board.push(move)

        score = stockfish_eval(board, engine) if engine else material_eval(board)
        return board.fen(), max(-15000, min(15000, score))
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description="Preprocess puzzles for NNUE training")
    parser.add_argument("-i", "--input", default="data/lichess_db_puzzle.csv")
    parser.add_argument("-o", "--output-dir", default="data")
    parser.add_argument("-s", "--stockfish", default=None, help="Path to stockfish")
    parser.add_argument("-n", "--max", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--toy", action="store_true", help="Generate toy dataset (10k each split)"
    )
    args = parser.parse_args()

    if args.toy:
        args.max = 35000  # ~10k each after 90/5/5 split, with some buffer for invalid
        args.output_dir = "data/toy"

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        print(
            "Download: wget https://database.lichess.org/lichess_db_puzzle.csv.zst && zstd -d *.zst"
        )
        sys.exit(1)

    # Setup stockfish
    engine = None
    if args.stockfish:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
            print(f"Using Stockfish: {args.stockfish}")
        except Exception as e:
            print(f"Stockfish failed ({e}), using material eval")

    # Count lines
    print("Counting lines...")
    with open(args.input) as f:
        total = sum(1 for _ in f) - 1
    if args.max:
        total = min(total, args.max)

    # Process
    print(f"Processing {total} puzzles...")
    data = []
    with open(args.input) as f:
        f.readline()  # skip header
        for i, line in enumerate(tqdm(f, total=total)):
            if args.max and i >= args.max:
                break
            result = process_puzzle(line, engine)
            if result:
                data.append(result)

    if engine:
        engine.quit()

    print(f"Valid: {len(data)}")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(data)

    n = len(data)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    n_train = n - n_val - n_test

    splits = {
        "train": data[:n_train],
        "val": data[n_train : n_train + n_val],
        "test": data[n_train + n_val :],
    }

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, rows in splits.items():
        df = pd.DataFrame(rows, columns=["fen", "score"])
        path = out_dir / f"{name}.csv.gz"
        df.to_csv(path, index=False, compression="gzip")
        print(f"{name}: {len(df)} -> {path}")

    # Stats
    all_scores = [s for _, s in data]
    print(
        f"\nScore stats: min={min(all_scores)}, max={max(all_scores)}, "
        f"mean={sum(all_scores) / len(all_scores):.0f}"
    )


if __name__ == "__main__":
    main()
