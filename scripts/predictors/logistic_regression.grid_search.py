#!/usr/bin/env python

'''
Make predictions for the test data
{'C': [0.001, 0.01, 0.1, 1.0, 10., 100.]}
'''

import argparse, logging
import cPickle as pickle
import numpy as np
from common import *
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.DEBUG)

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=load_encoded_features,
        help='Training features (npz)')
    parser.add_argument('labels', type=load_npz,
        help='Training labels (npz)')
    parser.add_argument('--classifiers', type=argparse.FileType('w'),
        help='Save fit classifiers (pkl)')
    parser.add_argument('--label', type=int, action='append',
        help='Labels to predict')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    Y = args.labels['labels']
    N = Y.shape[0]
    N_train = int(0.75 * N)
    logging.info("N_train = %d, N_test = %d" % (N_train, N-N_train))
    Y_train = Y[:N_train]
    Y_test = Y[N_train:]
    X_train = args.train[:N_train]
    X_test = args.train[N_train:]
    if args.label is None:
        args.label = np.arange(Y_train.shape[1])
    
    logging.info("Fitting classifiers")
    clfs = []
    for i in args.label:
        logging.info("Label %d" % i)
        clfsi = []
        for c in (0.75, 1.5, 5.):
            logging.info("C = %g" % c)
            clf = LogisticRegression(C=c, penalty='l1', tol=0.001, random_state=42)
            try:
                clf.fit(X_train, Y_train[:,i])
                print clf.coef_
                p = clf.predict_proba(X_test)
                s = score_predictions(Y_test[:,i], p[:,1])
                logging.info("...score = %f" % s)
                clfsi.append(clf)
            except Exception, e:
                logging.error(e)
                clfsi.append(None)    
        clfs.append(clfsi)        
        
    if args.classifiers:
        logging.info("Saving fit classifiers")
        pickle.dump(clfs, args.classifiers, 2)
