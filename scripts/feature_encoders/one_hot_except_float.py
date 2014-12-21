#!/usr/bin/env python

'''
Just the raw features
'''

import argparse, logging
import numpy as np
from scipy import sparse
from common import load_npz, save_encoded_features
from prepare_features import OneHotEncoder

logging.basicConfig(level=logging.DEBUG)

def one_hot(loader, output, cutoff=0, encoder=None):
    logging.info("Loading raw data")
    bfs = loader['bfeatures']
    ifs = loader['ifeatures']
    sfs = loader['sfeatures']
    X1 = np.hstack((bfs, ifs, sfs))
    del bfs, ifs, sfs
    X2 = sparse.csr_matrix(loader['ffeatures'])
    
    if encoder is None:
        logging.info("Making new one-hot encoder")
        encoder = OneHotEncoder()
        encoder.fit(X1, cutoff)
    logging.info("One-hot encoding data")
    X_hot = encoder.transform(X1)
    X = sparse.hstack((X_hot, X2))
    logging.info("Saving one-hotted data to output")
    save_encoded_features(output, X)
    return encoder

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
    enc = one_hot(args.train, args.train_out, args.cutoff)
    logging.info("One-hot encoding test data")
    one_hot(args.test, args.test_out, args.cutoff, enc)
