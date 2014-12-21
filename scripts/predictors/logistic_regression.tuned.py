#!/usr/bin/env python

'''
Make predictions for the test data
'''

import argparse, logging, multiprocessing
import cPickle as pickle
import numpy as np
from common import load_npz, save_npz, load_encoded_features
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.DEBUG)

CFG = {22: 1.0,
       23: 10.0,
       24: 10.0,
       25: 10.0,
       26: 10.0,
       27: 10.0,
       28: 0.75,
       29: 2.0,
       30: 1.0,
       31: 1.5,
       32: 0.75}
#CFG = {}

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=load_encoded_features,
        help='Training features (npz)')
    parser.add_argument('labels', type=load_npz,
        help='Training labels (npz)')
    parser.add_argument('test', type=load_encoded_features,
        help='Test features (npz)')
    parser.add_argument('output',
        help='Test predictions (npz)')
    parser.add_argument('meta',
        help='Train predictions (npz)')
    parser.add_argument('--col', type=int, action='append',
        help='Only predict certain columns')
    parser.add_argument('--fraction', type=float, default=0.5,
        help='Fit first fraction of data, predict for meta with rest')
    parser.add_argument('--invert', action='store_true',
        help='Fit second fraction of data')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    X = args.train
    Y = args.labels['labels']
    N = Y.shape[0]
    N1 = int(args.fraction * N)
    if args.invert: # for meta predictions
        X_train = X[N1:]
        Y_train = Y[N1:]
    else: # for first-round predictions
        X_train = X[:N1]
        Y_train = Y[:N1]
        
    logging.info("Fitting classifiers")
    Y_meta = []
    Y_test = []
    for i, y in enumerate(Y_train.T):
        if args.col and i not in args.col:
            y_hat = y.mean()
            Y_test.append(y_hat*np.ones(args.test.shape[0]))
            Y_meta.append(y_hat*np.ones(args.train.shape[0]))
            continue
            
        logging.info(i)
        C = CFG.get(i, 5.0)
        clf = LogisticRegression(C=C, tol=0.0001, random_state=42)
        if len(np.unique(y)) == 1:
            Y_test.append(y[0]*np.ones(args.test.shape[0]))
            Y_meta.append(y[0]*np.ones(args.train.shape[0]))
        else:
            logging.info("Fitting")
            clf.fit(X_train, y)
            logging.info("Predicting")
            p = clf.predict_proba(args.test)
            y = 1 - p[:,0]
            Y_test.append(y)
            p = clf.predict_proba(args.train)
            y = 1 - p[:,0]
            Y_meta.append(y)
            
    logging.info("Saving predictions to %s" % args.output)
    test = load_npz('../../data/test.npz')
    Y_test = np.vstack(Y_test).T
    save_npz(args.output, ids=test['ids'], 
        header=args.labels['header'], labels=Y_test)
    del Y_test, test
        
    logging.info("Saving predictions to %s" % args.meta)
    Y_meta = np.vstack(Y_meta).T
    save_npz(args.meta, ids=args.labels['ids'], 
        header=args.labels['header'], labels=Y_meta)
        
