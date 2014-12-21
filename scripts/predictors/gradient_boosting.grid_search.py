#!/usr/bin/env python

'''
Make predictions for the test data

        clf = GridSearchCV(rf, [{'criterion': ['entropy', 'gini'],
                                 'max_features': ['auto', 0.05, 0.1, 0.15, 0.2],
                                 'max_depth': [None],
                                 'min_samples_split': [1, 2],
                                 'min_samples_leaf': [1, 2],
                                 'max_leaf_nodes': [None]}], scoring=scorer, 
                           n_jobs=1, cv=3, verbose=2)
                           
0.45, 4: 0.08131
0.55, 4: 0.079448
'''

import argparse, logging, itertools
import cPickle as pickle
import numpy as np
from common import *
from sklearn.ensemble import GradientBoostingClassifier

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
        grid = itertools.product((0.2, 0.4, 0.6), (2, 4, 8, 16), (2, 3, 4, 5))
        for mf, msl, md in grid:
            logging.info("mf = %s, msl = %s" % (mf, msl))
            clf = GradientBoostingClassifier(
                loss='deviance', learning_rate=0.1, n_estimators=100,
                min_samples_split=1, max_depth=md, max_leaf_nodes=None,
                max_features=mf, min_samples_leaf=msl,
                verbose=2)
            try:
                clf.fit(X_train, Y_train[:,i])
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
