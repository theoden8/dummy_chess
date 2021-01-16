#!/usr/bin/env python3


import sys


s = f'{int(sys.argv[1]):064b}'
for i in range(8):
    y = 8 - i - 1
    y = i
    for j in range(8):
        x = 8 - j - 1
        sys.stdout.write(s[y*8 + x] + ' ')
    print()
