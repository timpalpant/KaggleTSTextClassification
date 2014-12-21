#!/usr/bin/env python

'''
Make predictions for the test data
'''

import argparse, logging
import cPickle as pickle
from common import load_npz, load_encoded_features, save_npz
from predictors import TSEnsembleClassifier

logging.basicConfig(level=logging.DEBUG)

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=load_encoded_features,
        help='Training features (npz)')
    parser.add_argument('labels', type=load_npz,
        help='Training labels (npz)')
    parser.add_argument('test', type=load_encoded_features,
        help='Test features (npz)')
    parser.add_argument('output',
        help='Output label predictions (npz)')
    parser.add_argument('--classifiers', type=argparse.FileType('w'),
        help='Save fit classifiers (pkl)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    
    logging.info("Preparing features")
    clf = TSEnsembleClassifier()
    logging.info("Fitting classifier(s)")
    clf.fit(args.train, args.labels['labels'])
    
    if args.classifiers:
        logging.info("Saving classifiers")
        pickle.dump(clf, args.classifiers)
    
    logging.info("Predicting for test data")
    Y = clf.predict(args.test)
    
    logging.info("Saving predictions to %s" % args.output)
    save_npz(args.output, ids=args.test['ids'], 
        header=args.labels['header'], labels=Y)