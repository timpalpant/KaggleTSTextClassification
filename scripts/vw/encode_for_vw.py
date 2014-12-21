#!/usr/bin/env python

'''
Put label on VW features
'''

import argparse, gzip
from itertools import izip
import numpy as np
from common import load_npz

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=gzip.GzipFile,
        help='VW features')
    parser.add_argument('labels', type=load_npz,
        help='Train labels (npz)')
    parser.add_argument('output',
        help='Output train features (vw)')
    parser.add_argument('--label', type=int, required=True,
        help='Label to write to output')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    print "Loading labels"
    y = np.array(args.labels['labels'][:,args.label], dtype=np.int8)
    y[y==0] = -1 # VW takes {-1, 1}
    print "Prepending labels onto VW features"
    with gzip.GzipFile(args.output, 'w') as out:
        for i, (x_i, y_i) in enumerate(izip(args.train, y)):
            if i % 10000 == 0: print i
            out.write("%d %s" % (y_i, x_i))
        
