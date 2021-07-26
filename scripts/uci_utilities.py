#!/usr/bin/env python3


import io
import os
import sys
import subprocess
import random
import time
import typing
from pprint import pprint


def get_file_content_str(filename):
    s = ""
    with open(filename, 'r') as f:
       for line in f:
           s += line
    return s


def get_output(command):
    tempfile = "tempfile"
    subprocess.call(command + " 2>&1 > " + tempfile, shell=True)
    s = get_file_content_str(tempfile)
    subprocess.call("rm -f " + tempfile, shell=True)
    return s


[STANDARD, CHESS960, CRAZYHOUSE] = range(3)
def get_setoption_uci(variant=STANDARD):
    if variant == STANDARD:
        return ''
    if variant == CHESS960:
        return f"echo 'setoption name UCI_Chess960 value true';"
    elif variant == CRAZYHOUSE:
        return f"echo 'setoption name UCI_Variant value crazyhouse';"
    return ''


class UCISession:
    startingpos = 'startpos'
    def __init__(self, variant=STANDARD) -> None:
        self.position_is_set = False
        self.commands: list = []
        self.set_variant(variant=variant)

    def set_variant(self, variant=STANDARD) -> None:
        if variant == STANDARD:
            pass
        elif variant == CHESS960:
            self.setoption(optname='UCI_Chess960', optvalue='true')
        elif variant == CRAZYHOUSE:
            self.setoption(optname='UCI_Variant', optvalue='crazyhouse')
        self.variant = variant

    def uciok(self):
        self.commands += ['uciok']

    def setoption(self, optname: str, optvalue):
        assert not self.position_is_set, 'please set position before setting options'
        self.commands += [f'setoption name {optname} value {optvalue}']

    def position(self, fen=None, moves=[]) -> None:
        sfen = ''
        if fen is None:
            sfen = UCISession.startingpos
        else:
            sfen = f'fen {fen}'
        self.commands += [f'position {sfen}' + '' if len(moves) == 0 else ' ' + ' '.join(moves)]
        self.position_is_set = True

    def display(self) -> None:
        self.commands += ['display']

    def go_perft(self, depth: int) -> None:
        if not self.position_is_set:
            self.position()
        self.commands += [f'go perft {depth}']

    def go(self, movetime=3., depth=1000) -> None:
        if not self.position_is_set:
            self.position()
        self.commands += [f'go depth {depth} movetime {int(movetime*1000)}']

    def stop(self) -> None:
        self.commands += ['stop']

    def special_command(self, command) -> None:
        self.commands += [command]

    def do_sleep(self, dur: float) -> None:
        self.commands += [['sleep', dur]]

    def do_expect(self, keyword: str) -> None:
        self.commands += [['expect', keyword]]

    def do_time(self) -> None:
        self.commands += [['time']]

    def run(self, uci_exec, info_func=print) -> typing.Any:
        active_time = time.time()
        p = subprocess.Popen(uci_exec, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        s = ''
        for cmd in self.commands:
            if type(cmd) is str:
                p.stdin.write(cmd + '\n')
                p.stdin.flush()
                #print('>', cmd)
            elif type(cmd) is list:
                if cmd[0] == 'sleep':
                    _, dur = cmd
                    #print('>> sleep', dur)
                    active_time = time.time() + dur
                    while time.time() < active_time:
                        continue
                elif cmd[0] == 'expect':
                    _, keyword = cmd
                    #print('>> expect', keyword)
                    for line in p.stdout:
                        s += line
                        info_func(line.strip())
                        if keyword in line:
                            #print('> break')
                            break
                elif cmd[0] == 'time':
                    yield time.time()
        #print('> close stdin')
        p.stdin.close()
        s += ''.join(p.stdout.readlines())
        p.wait()
        yield s


def noprint(s: str) -> None:
    pass


if __name__ == "__main__":
    sess = UCISession(variant=STANDARD)
    sess.go()
    sess.do_expect('bestmove')
    sess.stop()
    print(next(sess.run('./dummy_chess_uci')))
