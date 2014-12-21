#!/usr/bin/env python

'''
If p(y33) > cutoff, set y1:32 = 0
'''

import sys, argparse
import numpy as np
from common import *

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=load_npz,
        help='Input prediction probabilities (npz)')
    parser.add_argument('output',
        help='Output prediction probabilities (npz)')
    parser.add_argument('--cutoff', type=float, default=0.99,
        help='Minimum prediction probability')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    data = dict(args.input)
    labels = data['labels']
    y32 = (labels[:,32] > args.cutoff)
    print "%d rows with y33 > %f" % (y32.sum(), args.cutoff)
    mask = (labels[y32,:32] > 1e-2)
    labels[y32,:32] = 0
    data['labels'] = labels
    save_npz(args.output, **data)