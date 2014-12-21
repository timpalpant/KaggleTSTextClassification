#!/usr/bin/env python

'''
Just the raw features
'''

import argparse, logging
import numpy as np
from scipy import sparse
from common import load_npz, save_encoded_features
from sklearn.utils import murmurhash3_32

logging.basicConfig(level=logging.DEBUG)

def one_hot_hash(loader, output, d):
    logging.info("Loading raw data")
    bfs = loader['bfeatures']
    ffs = loader['ffeatures']
    ifs = loader['ifeatures']
    sfs = loader['sfeatures']
    X = np.hstack((bfs, ffs, ifs, sfs))
    del bfs, ffs, ifs, sfs
    
    nrows = X.shape[0]
    ncols = X.shape[1]
    ij = np.zeros((2, nrows*ncols), dtype=int) # row, col indices
    for i, row in enumerate(X):
        if i % 100000 == 0: logging.debug(i)
        start = i * ncols
        end = (i+1) * ncols
        ij[0,start:end] = i
        for j, x in enumerate(row):
            ij[1,start+j] = murmurhash3_32('%d_%s' % (j,x), seed=42, positive=True) % d
    data = np.ones(ij.shape[1]) # all ones
    X_hot = sparse.csr_matrix((data, ij), shape=(nrows, d))
    
    logging.info("Saving one-hotted data to output")
    save_encoded_features(output, X_hot)

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=load_npz,
        help='Training features (npz)')
    parser.add_argument('test', type=load_npz,
        help='Test features (npz)')
    parser.add_argument('train_out',
        help='Output train features (npz)')
    parser.add_argument('test_out',
        help='Output test features (npz)')
    parser.add_argument('-d', type=int, default=20,
        help='Dimension of hash table = 2**d')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    logging.info("One-hot encoding train data")
    one_hot_hash(args.train, args.train_out, 2**args.d)
    logging.info("One-hot encoding test data")
    one_hot_hash(args.test, args.test_out, 2**args.d)
