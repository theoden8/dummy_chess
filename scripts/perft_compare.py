#!/usr/bin/env python3


import os
import sys
import subprocess
import random
import time
from pprint import pprint


def get_file_content_str(filename):
    s = ""
    with open(filename) as f:
       for line in f:
           s += line
    return s


def get_output(command):
    tempfile = "tempfile"
    subprocess.call(command + " 2>&1 > " + tempfile, shell=True)
    s = get_file_content_str(tempfile)
    subprocess.call("rm -f " + tempfile, shell=True)
    return s


startingpos = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
def get_output_uci(uci_exec, depth: int, fen=startingpos) -> dict:
    start = time.time()
    s = get_output(f"(echo 'position fen {fen}'; echo 'go perft {depth}') | {uci_exec}")
    dur = time.time() - start
    print(f'{uci_exec + " " * (30 - len(uci_exec))}: {dur:.3f}s')
    turnmaps = {}
    for line in s.split('\n'):
        if ':' in line:
            left, right = line.split(':')
            left = left.replace('Nodes searched', 'total')
            turnmaps[left.strip()] = right.strip();
    return turnmaps


def get_output_stockfish(depth=5, fen=startingpos) -> dict:
    return get_output_uci('stockfish', depth=depth, fen=fen)


def get_output_dummy_chess(depth=5, fen=startingpos) -> dict:
    return get_output_uci('./dummy_chess_uci_opt', depth=depth, fen=fen)


def get_next_fen(fen, move):
    s = get_output(f"(echo 'position fen {fen} moves {move}'; echo 'd') | {stockfish} | grep Fen")
    s = s.replace('Fen: ', '')
    return s.strip()


def compare_outputs(depth=5, fen=startingpos, path=[]):
    if depth < 1:
        return True
    print(f"path={path}, depth={depth}, fen={fen}")
    sfmaps = get_output_stockfish(depth, fen)
    dcmaps = get_output_dummy_chess(depth, fen)
    print(f"totals: sf={sfmaps['total']}, dc={dcmaps['total']}")
    if sfmaps == dcmaps:
        return path == []
    flag_exit = False
    for k in sfmaps.keys():
        if k not in dcmaps:
            print(f'cannot find move {path + [k]}')
            flag_exit = True
    for k in dcmaps.keys():
        if k not in sfmaps:
            print(f'extra move in {path + [k]}')
            flag_exit = True
    if flag_exit:
        print(f"path={path}, depth={depth}, fen={fen}")
        return False
    diff = [k for k in sfmaps.keys() if sfmaps[k] != dcmaps[k] and k != 'total']
    random.shuffle(diff)
    for m in diff:
        res = compare_outputs(depth=depth-1, fen=get_next_fen(fen, m), path=path+[m])
    return False


if __name__ == "__main__":
    # https://github.com/official-stockfish/Stockfish/blob/master/tests/perft.sh
    # https://www.chessprogramming.org/Perft_Results
    compare_outputs(depth=6, fen=startingpos)
    compare_outputs(depth=5, fen='r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 0')
    compare_outputs(depth=6, fen='8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 0')
    compare_outputs(depth=5, fen='r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1')
    compare_outputs(depth=5, fen='rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8')
    compare_outputs(depth=5, fen='r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10')
    compare_outputs(depth=2, fen='r2qk1nr/2pn2Pp/8/8/2P5/8/4PP1P/1Nb1KBNb w kq - 0 10')
    compare_outputs(depth=2, fen='8/kp4P1/p3K3/4P3/3p1P2/4nr2/8/8 w - - 1 55')
