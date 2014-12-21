#!/usr/bin/env python

'''
Transform prediction probabilities
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
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    data = dict(args.input)
    labels = data['labels']
    labels[labels < 0.5] = labels[labels < 0.5]**2
    labels[labels > 0.5] = 1 - (1-labels[labels > 0.5])**2
    data['labels'] = labels
    save_npz(args.output, **data)