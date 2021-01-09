#!/usr/bin/env python3


import os
import sys
import subprocess
import random
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
stockfish = 'stockfish'
def get_output_stockfish(depth=5, fen=startingpos):
    s = get_output(f"(echo 'position fen {fen}'; echo 'go perft {depth}') | {stockfish}")
    turnmaps = {}
    for line in s.split('\n'):
        if ':' in line:
            left, right = line.split(':')
            left = left.replace('Nodes searched', 'total')
            turnmaps[left.strip()] = right.strip();
    return turnmaps


def get_next_fen(fen, move):
    s = get_output(f"(echo 'position fen {fen} moves {move}'; echo 'd') | {stockfish} | grep Fen")
    s = s.replace('Fen: ', '')
    return s.strip()


def get_output_dummy_chess(depth=5, fen=startingpos):
    perft = './dummy_chess_perft'
    s = get_output(f'{perft} {depth} "{fen}"')
    turnmaps = {}
    for line in s.split('\n'):
        if ':' in line:
            left, right = line.split(':')
            turnmaps[left.strip()] = right.strip();
    return turnmaps


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
    #compare_outputs(depth=5, fen=startingpos)
    #compare_outputs(depth=4, fen='r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 0')
    compare_outputs(depth=3, fen='8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 0')
