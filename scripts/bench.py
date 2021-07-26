#!/usr/bin/env python3


from uci_utilities import *


uci_exec = './dummy_chess_uci'


def perft_output_map(s: str) -> dict:
    turnmaps = {}
    for line in s.split('\n'):
        if ':' in line:
            left, right = line.split(':')
            left = left.replace('Nodes searched', 'total')
            turnmaps[left.strip()] = right.strip()
    return turnmaps


def bench_perft_fen(maxdepth, variant=STANDARD, fen=None, shannon=None):
    print('fen', fen if fen is not None else 'startpos')
    for depth in range(1, min(maxdepth, len(shannon if shannon is not None else maxdepth)) + 1):
        sess = UCISession(variant=variant)
        sess.setoption(optname='Hash', optvalue=1024)
        sess.position(fen)
        sess.do_time()
        sess.go_perft(depth=depth)
        sess.display()
        sess.do_expect('Nodes')
        sess.do_time()
        gen = sess.run(uci_exec, info_func=noprint)
        start = next(gen)
        stop = next(gen)
        s = next(gen)
        # stats
        dur = stop - start + 1e-9
        meta_info = perft_output_map(s)
        nodes = int(meta_info['total'])
        hashfull = float(meta_info['display stat_hashfull'])
        hit_rate = float(meta_info['display stat_hit_rate'])
        nodes_searched = int(meta_info['display stat_nodes_searched'])
        shannon_val = shannon[depth-1] if shannon is not None else ''
        print(f'depth={depth} time={dur:.3f}\t{nodes/dur/1e3:.3f} kN/sec\traw={nodes_searched/dur/1e3:.3f} kN/sec\t'
                f'nodes={nodes}\tshannon={shannon_val}\t'
                f'hit_rate={hit_rate:.3f}\thashfull={hashfull:.3f}')
    print()


def bench_perft():
    print('perft benchmarks')
    bench_perft_fen(maxdepth=7, shannon=[20, 400, 8902, 197281, 4865609, 119060324, 3195901860, 84998978956, 2439530234167])
    bench_perft_fen(maxdepth=5, fen='r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ',
                                   shannon=[48, 2039, 97862, 4085603, 193690690, 8031647685])


if __name__ == "__main__":
    bench_perft()
