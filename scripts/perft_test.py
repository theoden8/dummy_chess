#!/usr/bin/env python3


from uci_utilities import *


def get_output_uci(uci_exec, depth: int, fen=None, variant=STANDARD) -> dict:
    sess = UCISession(variant=variant)
    sess.position(fen)
    sess.do_time()
    sess.go_perft(depth=depth)
    sess.do_expect('Nodes')
    sess.do_time()
    gen = sess.run(uci_exec, info_func=noprint)
    start = next(gen)
    dur = next(gen) - start
    s = next(gen)
    print(f'{uci_exec + " " * (30 - len(uci_exec))}: {dur:.3f}s')
    turnmaps = {}
    for line in s.split('\n'):
        if ':' in line:
            left, right = line.split(':')
            left = left.replace('Nodes searched', 'total')
            turnmaps[left.strip()] = right.strip()
    return turnmaps


def get_uciexec(variant=STANDARD) -> str:
    if variant in [STANDARD, CHESS960]:
        return 'stockfish'
    return 'fairy-stockfish'


def get_output_stockfish(depth=5, fen=None, variant=STANDARD) -> dict:
    return get_output_uci(get_uciexec(variant=variant), depth=depth, fen=fen, variant=variant)


def get_output_dummy_chess(depth=5, fen=None, variant=STANDARD) -> dict:
    return get_output_uci('./dummy_chess_uci', depth=depth, fen=fen, variant=variant)


def get_next_fen(fen: str, move: str, variant=STANDARD) -> str:
    uciexec = get_uciexec(variant=variant)
    sess = UCISession(variant=variant)
    sess.position(fen=fen, moves=[move])
    sess.special_command('d')
    s = next(sess.run(uciexec))
    s = ''.join([ss for ss in s.split('\n') if 'Fen' in ss])
    s = s.replace('Fen: ', '')
    return s.strip()


def compare_outputs(depth=5, fen=None, path=[], variant=STANDARD) -> bool:
    if depth < 1:
        return True
    print(f"path={path}, depth={depth}, fen={fen}")
    sfmaps = get_output_stockfish(depth, fen, variant=variant)
    dcmaps = get_output_dummy_chess(depth, fen, variant=variant)
    print(f"totals: sf={sfmaps['total']}, dc={dcmaps['total']}")
    if sfmaps == dcmaps:
        return path == []
    if sfmaps['total'] == dcmaps['total']:
        return path == []
    assert sfmaps['total'] == dcmaps['total'], f"different results ({sfmaps['total']}, {dcmaps['total']})"
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
    diff = [k for k in dcmaps.keys() if sfmaps[k] != dcmaps[k] and k != 'total']
    for m in diff:
        res = compare_outputs(depth=depth-1, fen=get_next_fen(fen=fen, move=m, variant=variant), path=path+[m], variant=variant)
    return False


def compare_outputs_960(depth: int, fen: str, path=[]) -> bool:
    return compare_outputs(depth=depth, fen=fen, path=path, variant=CHESS960)


def compare_outputs_ch(depth: int, fen: str, path=[]) -> bool:
    return compare_outputs(depth=depth, fen=fen, path=path, variant=CRAZYHOUSE)


def check_traditional() -> None:
    # https://github.com/official-stockfish/Stockfish/blob/master/tests/perft.sh
    # https://www.chessprogramming.org/Perft_Results
    compare_outputs(depth=6, fen=None)
    compare_outputs(depth=5, fen='r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 0')
    compare_outputs(depth=6, fen='8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 0')
    compare_outputs(depth=5, fen='r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1')
    compare_outputs(depth=5, fen='rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8')
    compare_outputs(depth=5, fen='r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10')
    compare_outputs(depth=5, fen='r2qk1nr/2pn2Pp/8/8/2P5/8/4PP1P/1Nb1KBNb w kq - 0 10')
    compare_outputs(depth=5, fen='8/kp4P1/p3K3/4P3/3p1P2/4nr2/8/8 w - - 1 55')
    compare_outputs(depth=4, fen='q2k2q1/2nqn2b/1n1P1n1b/2rnr2Q/1NQ1QN1Q/3Q3B/2RQR2B/Q2K2Q1 w - -')


def check_chess960() -> None:
    compare_outputs_960(depth=1, fen='4k3/8/8/8/8/8/8/rR2K1N1 w Q - 0 1')
    compare_outputs_960(depth=5, fen='r2k2rB/ppp2b1p/3n3b/3p1pp1/4p3/P2P2P1/1PP1PPBP/RN3RKQ b kq - 0 11')
    # https://www.chessprogramming.org/Chess960_Perft_Results
    # 0
    compare_outputs_960(depth=5, fen='bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9')
    compare_outputs_960(depth=5, fen='2nnrbkr/p1qppppp/8/1ppb4/6PP/3PP3/PPP2P2/BQNNRBKR w HEhe - 1 9')
    compare_outputs_960(depth=5, fen='b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9')
    compare_outputs_960(depth=5, fen='qbbnnrkr/2pp2pp/p7/1p2pp2/8/P3PP2/1PPP1KPP/QBBNNR1R w hf - 0 9')
    compare_outputs_960(depth=5, fen='1nbbnrkr/p1p1ppp1/3p4/1p3P1p/3Pq2P/8/PPP1P1P1/QNBBNRKR w HFhf - 0 9')
    compare_outputs_960(depth=5, fen='qnbnr1kr/ppp1b1pp/4p3/3p1p2/8/2NPP3/PPP1BPPP/QNB1R1KR w HEhe - 1 9')
    compare_outputs_960(depth=5, fen='q1bnrkr1/ppppp2p/2n2p2/4b1p1/2NP4/8/PPP1PPPP/QNB1RRKB w ge - 1 9')
    compare_outputs_960(depth=5, fen='qbn1brkr/ppp1p1p1/2n4p/3p1p2/P7/6PP/QPPPPP2/1BNNBRKR w HFhf - 0 9')
    compare_outputs_960(depth=5, fen='qnnbbrkr/1p2ppp1/2pp3p/p7/1P5P/2NP4/P1P1PPP1/Q1NBBRKR w HFhf - 0 9')
    compare_outputs_960(depth=5, fen='qn1rbbkr/ppp2p1p/1n1pp1p1/8/3P4/P6P/1PP1PPPK/QNNRBB1R w hd - 2 9')
    # 10
    compare_outputs_960(depth=5, fen='qnr1bkrb/pppp2pp/3np3/5p2/8/P2P2P1/NPP1PP1P/QN1RBKRB w GDg - 3 9')
    compare_outputs_960(depth=5, fen='qb1nrkbr/1pppp1p1/1n3p2/p1B4p/8/3P1P1P/PPP1P1P1/QBNNRK1R w HEhe - 0 9')
    compare_outputs_960(depth=5, fen='qnnbrk1r/1p1ppbpp/2p5/p4p2/2NP3P/8/PPP1PPP1/Q1NBRKBR w HEhe - 0 9')
    compare_outputs_960(depth=5, fen='1qnrkbbr/1pppppp1/p1n4p/8/P7/1P1N1P2/2PPP1PP/QN1RKBBR w HDhd - 0 9')
    compare_outputs_960(depth=5, fen='qn1rkrbb/pp1p1ppp/2p1p3/3n4/4P2P/2NP4/PPP2PP1/Q1NRKRBB w FDfd - 1 9')
    compare_outputs_960(depth=5, fen='bb1qnrkr/pp1p1pp1/1np1p3/4N2p/8/1P4P1/P1PPPP1P/BBNQ1RKR w HFhf - 0 9')
    compare_outputs_960(depth=5, fen='bnqbnr1r/p1p1ppkp/3p4/1p4p1/P7/3NP2P/1PPP1PP1/BNQB1RKR w HF - 0 9')
    compare_outputs_960(depth=5, fen='bnqnrbkr/1pp2pp1/p7/3pP2p/4P1P1/8/PPPP3P/BNQNRBKR w HEhe d6 0 9')
    compare_outputs_960(depth=5, fen='b1qnrrkb/ppp1pp1p/n2p1Pp1/8/8/P7/1PPPP1PP/BNQNRKRB w GE - 0 9')
    compare_outputs_960(depth=5, fen='n1bqnrkr/pp1ppp1p/2p5/6p1/2P2b2/PN6/1PNPPPPP/1BBQ1RKR w HFhf - 2 9')
    # 20
    compare_outputs_960(depth=5, fen='n1bb1rkr/qpnppppp/2p5/p7/P1P5/5P2/1P1PPRPP/NQBBN1KR w Hhf - 1 9')
    compare_outputs_960(depth=5, fen='nqb1rbkr/pppppp1p/4n3/6p1/4P3/1NP4P/PP1P1PP1/1QBNRBKR w HEhe - 1 9')
    compare_outputs_960(depth=5, fen='n1bnrrkb/pp1pp2p/2p2p2/6p1/5B2/3P4/PPP1PPPP/NQ1NRKRB w GE - 2 9')
    compare_outputs_960(depth=5, fen='nbqnbrkr/2ppp1p1/pp3p1p/8/4N2P/1N6/PPPPPPP1/1BQ1BRKR w HFhf - 0 9')
    compare_outputs_960(depth=5, fen='nq1bbrkr/pp2nppp/2pp4/4p3/1PP1P3/1B6/P2P1PPP/NQN1BRKR w HFhf - 2 9')
    compare_outputs_960(depth=5, fen='nqnrb1kr/2pp1ppp/1p1bp3/p1B5/5P2/3N4/PPPPP1PP/NQ1R1BKR w HDhd - 0 9')
    compare_outputs_960(depth=5, fen='nqn2krb/p1prpppp/1pbp4/7P/5P2/8/PPPPPKP1/NQNRB1RB w g - 3 9')
    compare_outputs_960(depth=5, fen='nb1n1kbr/ppp1rppp/3pq3/P3p3/8/4P3/1PPPRPPP/NBQN1KBR w Hh - 1 9')
    compare_outputs_960(depth=5, fen='nqnbrkbr/1ppppp1p/p7/6p1/6P1/P6P/1PPPPP2/NQNBRKBR w HEhe - 1 9')
    compare_outputs_960(depth=5, fen='nq1rkb1r/pp1pp1pp/1n2bp1B/2p5/8/5P1P/PPPPP1P1/NQNRKB1R w HDhd - 2 9')
    # 30
    compare_outputs_960(depth=5, fen='nqnrkrb1/pppppp2/7p/4b1p1/8/PN1NP3/1PPP1PPP/1Q1RKRBB w FDfd - 1 9')
    compare_outputs_960(depth=5, fen='bb1nqrkr/1pp1ppp1/pn5p/3p4/8/P2NNP2/1PPPP1PP/BB2QRKR w HFhf - 0 9')
    compare_outputs_960(depth=5, fen='bnn1qrkr/pp1ppp1p/2p5/b3Q1p1/8/5P1P/PPPPP1P1/BNNB1RKR w HFhf - 2 9')
    compare_outputs_960(depth=5, fen='bnnqrbkr/pp1p2p1/2p1p2p/5p2/1P5P/1R6/P1PPPPP1/BNNQRBK1 w Ehe - 0 9')
    compare_outputs_960(depth=5, fen='b1nqrkrb/2pppppp/p7/1P6/1n6/P4P2/1P1PP1PP/BNNQRKRB w GEge - 0 9')
    compare_outputs_960(depth=5, fen='n1bnqrkr/3ppppp/1p6/pNp1b3/2P3P1/8/PP1PPP1P/NBB1QRKR w HFhf - 1 9')
    compare_outputs_960(depth=5, fen='n2bqrkr/p1p1pppp/1pn5/3p1b2/P6P/1NP5/1P1PPPP1/1NBBQRKR w HFhf - 3 9')
    compare_outputs_960(depth=5, fen='nnbqrbkr/1pp1p1p1/p2p4/5p1p/2P1P3/N7/PPQP1PPP/N1B1RBKR w HEhe - 0 9')
    compare_outputs_960(depth=5, fen='nnbqrkr1/pp1pp2p/2p2b2/5pp1/1P5P/4P1P1/P1PP1P2/NNBQRKRB w GEge - 1 9')
    compare_outputs_960(depth=5, fen='nb1qbrkr/p1pppp2/1p1n2pp/8/1P6/2PN3P/P2PPPP1/NB1QBRKR w HFhf - 0 9')


def check_crazyhouse() -> None:
    compare_outputs_ch(depth=6, fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/ w KQkq - 0 1')
    compare_outputs_ch(depth=4, fen='r1bq1rk1/pppp1p2/2n2p1p/2b1p3/2B1P3/2NP1N2/PPP2PPP/R2Q1RK1/Nb b - - 1 8')
    compare_outputs_ch(depth=4, fen='r2q2k1/p1Pp1r2/bpn2p1p/2b1p3/2B4N/PPN4P/2PbpPP1/R2Q1RK1/PNp w - - 0 17')
    compare_outputs_ch(depth=4, fen='rq2kb1N/2b3p1/2Pp2Pp/1P1Pp3/bP2p1b1/PPnPP1P1/4P2R/1R1QK3/RNN w q - 0 44')
    compare_outputs_ch(depth=4, fen='1r~4k1/p2P1N1R/1ppRpn1B/3p4/q2P4/1NP5/PP2bPPQ/R5K1/Bpppnbq w - - 5 43')
    # https://codeberg.org/theoden8/dummy_chess/issues/9
    compare_outputs_ch(depth=4, fen='1r1Q~1k2/p4NpR/1ppRpn1B/3p4/q2P4/1NP5/PP2bPPQ/R5K1/Bppnbq b - - 0 43')


if __name__ == "__main__":
    check_crazyhouse()
    check_chess960()
    check_traditional()
