#!/usr/bin/env python

'''
Just the raw features
'''

import argparse, logging, glob
import numpy as np
from scipy import sparse
from common import *
from collections import defaultdict

logging.basicConfig(level=logging.DEBUG)

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', type=glob.glob,
        help='Split features (npz)')
    parser.add_argument('output',
        help='Split features (npz)')
    parser.add_argument('-d', type=int, default=20,
        help='Reduce dimension to 2**d (set to 0 for original dimension)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    logging.info("Loading split data")
    i = [int(fn.split('.')[-4]) for fn in args.split]
    idx = np.argsort(i)
    args.split = list(np.take(args.split, idx))

    i = list()
    j = list()
    nrows = 0
    ncols = 2**args.d
    print "ncols = %d" % ncols
    for input_file in args.split:
        logging.info(input_file)
        X_i = load_encoded_features(input_file).tocoo()
        nrows += X_i.shape[0]
        if ncols==1: ncols = X_i.shape[1]
        print repr(X_i)
        i.append(np.asarray(X_i.row, dtype=np.uint32))
        j.append(np.asarray(X_i.col % ncols, dtype=np.uint32))
        del X_i
    logging.info("Merging row and column indices")
    row = np.asarray(np.concatenate(i), dtype=np.uint32)
    col = np.asarray(np.concatenate(j), dtype=np.uint32)
    del i, j
    logging.info("Generating data matrix")
    data = np.ones(row.shape[0], dtype=np.uint8)
    logging.info("Making CSR matrix")
    X = sparse.csr_matrix((data, (row,col)), shape=(nrows,ncols), 
                          dtype=np.float64)

    logging.info("Saving to output %s" % args.output)
    save_encoded_features(args.output, X)   
