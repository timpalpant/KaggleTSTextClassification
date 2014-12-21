#!/usr/bin/env python

'''
Make predictions for the test data
'''

import argparse, logging
import numpy as np
import cPickle as pickle
from common import load_npz, load_encoded_features, cross_validate
from predictors import TSEnsembleClassifier

logging.basicConfig(level=logging.DEBUG)

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=load_npz,
        help='Training features (npz)')
    parser.add_argument('labels', type=load_npz,
        help='Training labels (npz)')
    parser.add_argument('--cv', type=int, default=3,
        help='Number of cross-validation folds (default: %(default)d)')
    parser.add_argument('--classifiers', type=argparse.FileType('w'),
        help='Save fit classifiers (pkl)')
    parser.add_argument('--label', type=int, action='append',
        help='Only predict one label (default: all)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    
    clf = TSEnsembleClassifier()
    labels = args.labels['labels']
    if args.label: labels = labels[:,args.label]
    scores = cross_validate(clf, args.train, labels, n_folds=args.cv)
    print "Average log-loss for each label:"
    print scores.mean(axis=0)
    i = np.argsort(scores.mean(axis=0))[::-1]
    print "Labels from worst to best:"
    print i
    print "Average log-loss overall: %f (+/-) %f" \
        % (scores.mean(), scores.mean(axis=1).std())
    if args.classifiers:
        logging.info("Saving classifiers")
        pickle.dump(clf, args.classifiers)