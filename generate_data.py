"""
We are encoding out inputs with binary values
from 1 to 2^10
"""
import numpy as np

def labels(first, last):
    steps = last-first+1
    fb = []
    for i in np.linspace(first, last, steps):
        if i%15==0:
            fb.append([0,0,0,1])
        elif i%3==0:
            fb.append([0,1,0,0])
        elif i%5==0:
            fb.append([0,0,1,0])
        else:
            fb.append([1,0,0,0])
    return np.array(fb)

def inputs(first, last):
    return np.array(encode(range(first, last+1)))

def encode(inputs):
    results = []
    for inp in inputs:
        results.append(
            [int(inp) >> i & 1 for i in range(10)]
        )
    return results