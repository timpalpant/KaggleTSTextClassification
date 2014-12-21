#!/usr/bin/env python

'''
Just the raw features
'''

import argparse, logging, gc, objgraph
from itertools import izip
import numpy as np
from scipy import sparse
from common import load_npz, save_encoded_features
from prepare_features import OneHotEncoder

logging.basicConfig(level=logging.DEBUG)

def pair_features(to_pairs, pair_to_id={}):
    nrows = to_pairs.shape[0]
    ncols = to_pairs.shape[1]
    npairs = ncols * (ncols-1) / 2
    pairs = np.zeros((nrows, npairs), dtype=np.uint32)
    nunique = 0
    for i in xrange(nrows):
        if i % 50000 == 0: 
            gc.collect()
            #objgraph.show_most_common_types(limit=20)
            logging.debug(i)
        m = 0
        for j in xrange(ncols):
            row_pairs_to_id = pair_to_id.get(to_pairs[i,j], None)
            if row_pairs_to_id is None:
                row_pairs_to_id = {}
                pair_to_id[to_pairs[i,j]] = row_pairs_to_id
            for k in xrange(j):
                v = row_pairs_to_id.get(to_pairs[i,k], None)
                if v is None:
                    v = nunique
                    row_pairs_to_id[to_pairs[i,k]] = v
                    nunique += 1
                pairs[i,m] = v
                m += 1
    return pairs

def one_hot(loader, output, cutoff=0, encoder=None, all_pairs=None):
    logging.info("Loading raw data")
    sfs = loader['sfeatures']
    logging.info("Generating pair features")
    to_pairs = sfs #np.hstack((bfs,sfs))
    if all_pairs is None:
        all_pairs = dict()
    pairs = pair_features(to_pairs, all_pairs)
    ffs = loader['ffeatures']
    ifs = loader['ifeatures']
    bfs = loader['bfeatures']
    logging.info("Concatenating all features")
    X = np.hstack((to_pairs, bfs, ffs, ifs, pairs))
    del to_pairs, bfs, ffs, ifs, sfs, pairs
    
    if encoder is None:
        logging.info("Making new one-hot encoder")
        encoder = OneHotEncoder()
        encoder.fit(X, cutoff)
    logging.info("One-hot encoding data")
    X_hot = encoder.transform(X)
    del X
    logging.info("Saving one-hotted data to output")
    save_encoded_features(output, X_hot)
    return encoder, all_pairs

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
    parser.add_argument('--cutoff', type=int, default=10,
        help='Frequency cutoff')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    logging.info("One-hot encoding train data")
    enc, all_pairs = one_hot(args.train, args.train_out, args.cutoff)
    logging.info("One-hot encoding test data")
    one_hot(args.test, args.test_out, args.cutoff, enc, all_pairs)
