#!/usr/bin/env python3


import os
import re
import subprocess
import itertools

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_file_content_str(filename) -> str:
    s = ""
    with open(filename, 'r') as f:
       for line in f:
           s += line
    return s


if __name__ == "__main__":
    samples = []
    for i in range(4):
        tempfile = "tempfile"
        subprocess.call(f"./scripts/selfplay | tee {tempfile}", shell=True)
        s = get_file_content_str(tempfile)
        moves = [m for m in s.split()[:-1] if re.match('[0-9]+\.', m) is None]
        samples.append(moves)
        os.remove(tempfile)
    opening_streaks = np.zeros((len(samples), len(samples)))
    for i, j in itertools.combinations_with_replacement(range(len(samples)), 2):
        if i == j:
            continue
        val = 0
        for k in range(min(len(samples[i]), len(samples[j]))):
            if samples[i][k] != samples[j][k]:
                break
            val += 1
        opening_streaks[i,j] = val
        opening_streaks[j,i] = val
    plt.figure(figsize=(8, 8))
    plt.title('common openings')
    sns.heatmap((1 + opening_streaks) / 2, square=True, vmin=0, linewidth=.2, linecolor='w')
    plt.tight_layout(pad=.3)
    plt.show()
    plt.close()
