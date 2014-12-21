#!/usr/bin/env python

'''
Join predictions from other classifiers
to make meta-features.
'''

import argparse, logging
import numpy as np
from common import *

logging.basicConfig(level=logging.DEBUG)

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',
        help='Output features (npz)')
    parser.add_argument('--pred', type=load_npz, 
        action='append', required=True,
        help='Predictions to join (npz)')
    parser.add_argument('--feat', type=load_encoded_features, 
        action='append', default=[],
        help='Features to join (npz)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    logging.info("Loading predictions")
    X = [loader['labels'] for loader in args.pred]
    X += args.feat
    X = np.hstack(X)
    logging.info("Saving meta-features to output")
    save_encoded_features(args.output, X)    
