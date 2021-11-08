#!/usr/bin/env python3


import os
import sys
import tqdm

import pandas as pd

import chess
import dummy_chess


def wc_l(filename):
    i = 0
    with open(filename, 'r') as f:
        for line in f:
            i += 1
    return i


def make_puzzle_dataset(input_file: str):
    this_file = os.path.abspath(os.path.dirname(sys.argv[0]))
    to_here_path = os.path.relpath('.', this_file)
    filename = os.path.join(to_here_path, input_file)
    line_count = wc_l(filename)
    with open(filename, 'r') as f:
        pbar = tqdm.tqdm(total=line_count, desc=filename)
        df_fens, df_scores = [], []
        board = chess.Board()
        dummy = dummy_chess.ChessDummy()
        for line in f:
            puzzleid, fen, moves, rating, ratingdev, popularity, nbplays, themes, url = line.split(',')
            board.set_fen(fen)
            for m in moves.split():
                board.push(chess.Move.from_uci(m))
            dummy.set_fen(board.fen())
            score = dummy.evaluate()
            df_fens += [fen]
            df_scores += [score]
            pbar.update(1)
        df = pd.DataFrame(data=dict(fen=df_fens, score=df_scores))
        output_file = 'data/puzzle_dataset.csv.gz'
        df.to_csv(os.path.join(to_here_path, output_file), mode='w', sep=",",
                  index=False, compression='gzip')
    return output_file


def make_features_dataset(input_file: str):
    this_file = os.path.abspath(os.path.dirname(sys.argv[0]))
    to_here_path = os.path.relpath('.', this_file)
    df = pd.read_csv(os.path.join(to_here_path, input_file), sep=',', compression='gzip')
    df_scores = df.score
    df_feat1 = []
    df_feat2 = []
    nnue = dummy_chess.NNUEDummy()
    for fen in tqdm.tqdm(df.fen):
        nnue.set_fen(fen)
        sparse = nnue.halfkp
        df_feat1 += [sparse[0].tolist()]
        df_feat2 += [sparse[1].tolist()]
    output_file = 'data/sparse_dataset.h5'
    df = pd.DataFrame(data=dict(score=df_scores, sparse1=df_feat1, sparse2=df_feat2))
    df.to_hdf(os.path.join(to_here_path, output_file), key='sparse', mode='w', index=False)


if __name__ == "__main__":
    puzzles_file = make_puzzle_dataset('data/lichess_db_puzzle.csv')
    features_file = make_features_dataset(puzzles_file)
