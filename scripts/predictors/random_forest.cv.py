#!/usr/bin/env python

'''
Make predictions for the test data

TODO: Should explore min_samples_leaf >= 2 and max_features >= 0.2
'''

import argparse, logging
import cPickle as pickle
import numpy as np
from common import *
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.DEBUG)

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=load_encoded_features,
        help='Training features (npz)')
    parser.add_argument('labels', type=load_npz,
        help='Training labels (npz)')
        
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    Y_train = args.labels['labels']
    
    for i, y in enumerate(Y_train.T):
        logging.info(i)
        clf = RandomForestClassifier(
            n_estimators=48, max_features=0.2,
            min_samples_split=1, max_depth=None, max_leaf_nodes=None,
            criterion='entropy', min_samples_leaf=2,
            n_jobs=-1, random_state=42, verbose=2)
        try:
            scores = cross_validate(clf, args.train, Y_train)
        except Exception, e:
            logging.error(e)