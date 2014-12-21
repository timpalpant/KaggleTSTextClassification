#!/usr/bin/env python

'''
Just the raw features
'''

import argparse, logging
import numpy as np
from common import load_npz, save_encoded_features

logging.basicConfig(level=logging.DEBUG)

def raw_encode(loader, output):
    logging.info("Loading data")
    bfs = loader['bfeatures']
    ffs = loader['ffeatures']
    ifs = loader['ifeatures']
    sfs = loader['sfeatures']
    X = np.hstack((bfs, ffs, ifs, sfs))
    logging.info("Saving data")
    save_encoded_features(output, X)

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=load_npz,
        help='Train features (npz)')
    parser.add_argument('test', type=load_npz,
        help='Test features (npz)')
    parser.add_argument('train_out',
        help='Output train features (npz)')
    parser.add_argument('test_out',
        help='Output test features (npz)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    logging.info("Encoding train data")
    raw_encode(args.train, args.train_out)
    logging.info("Encoding test data")
    raw_encode(args.test, args.test_out)    