#!/usr/bin/env python

'''
Predict y33 = max(1e-2, 1 - sum(y1:y32))
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
    parser.add_argument('--min', type=float, default=1e-2,
        help='Minimum prediction probability')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    data = dict(args.input)
    labels = data['labels']
    y32 = 1 - labels[:,:32].sum(axis=1)
    y32[y32 < args.min] = args.min
    labels[:,32] = y32
    data['labels'] = labels
    save_npz(args.output, **data)