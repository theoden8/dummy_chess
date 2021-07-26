#!/usr/bin/env python3


import sys

from uci_utilities import *


def run_fen(sess: UCISession, movetime=5., fen=None) -> UCISession:
    sess.do_print('PGO', 'fen', fen if fen is not None else 'startpos')
    sess.position(fen)
    sess.go(movetime=movetime)
    sess.do_expect('bestmove')
    return sess


def run_traditional_pgo(uci_exec: str) -> None:
    sess = UCISession(variant=STANDARD)
    sess.setoption('Hash', 1024)
    run_fen(sess, fen='1r4k1/1r3pp1/3b3p/3p1qnP/Q1pP3R/2P2PP1/PP4K1/R1B3N1 b - - 2 24')
    run_fen(sess, fen='r1b1kb1r/pp2pp2/n1p1q1p1/1N1nN2p/2BP4/4BQ2/PPP2PPP/R4RK1 b kq - 1 11')
    run_fen(sess, fen='8/5k2/2pBp2p/6p1/pP2P3/P1R1K2P/2P5/3r4 w - - 3 49')
    run_fen(sess, fen='rnq1kbnr/p1p4p/1p2pp2/3p2p1/2PP3N/4P3/PP1B1PQP/RN2K2R w KQkq - 0 11')
    run_fen(sess, fen='2rqkb1r/p2b1pp1/2n5/1p1n2Pp/N3p2P/1PPp4/P2P1P2/R1BQKBNR w KQk - 0 15')
    run_fen(sess, fen='r3kb1r/p1p1ppp1/pq2Nn1p/4N3/3P1B2/8/PPP2PPP/R2Q1RK1 b kq - 0 12')
    next(sess.run('./dummy_chess_uci', noprint))


if __name__ == "__main__":
    run_traditional_pgo(sys.argv[1])
