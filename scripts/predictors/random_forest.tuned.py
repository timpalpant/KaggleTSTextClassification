#!/usr/bin/env python

'''
Make predictions for the test data
'''

import argparse, logging
import cPickle as pickle
import numpy as np
from common import load_npz, save_npz, load_encoded_features
from sklearn.ensemble import RandomForestClassifier

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
        help='Test predictions (npz)')
    parser.add_argument('--col', type=int, action='append',
        help='Only predict certain columns')
    parser.add_argument('--classifiers', type=argparse.FileType('w'),
        help='Save fit classifiers to Pickle file')
    return parser
    
CFG = [{'max_features': 0.2,
  'min_samples_leaf': 2},
 {'max_features': 'auto',
  'min_samples_leaf': 2},
 {'max_features': 0.2,
  'min_samples_leaf': 2},
 {'max_features': 'auto',
  'min_samples_leaf': 2},
 {'max_features': 0.15,
  'min_samples_leaf': 2},
 {'max_features': 0.2,
  'min_samples_leaf': 2},
 {'max_features': 'auto',
  'min_samples_leaf': 2},
 {'max_features': 'auto',
  'min_samples_leaf': 2},
 {'max_features': 0.2,
  'min_samples_leaf': 2},
 {'max_features': 0.05,
  'min_samples_leaf': 2},
 {'max_features': 0.2,
  'min_samples_leaf': 2},
 {'max_features': 0.2,
  'min_samples_leaf': 2},
 {'max_features': 'auto',
  'min_samples_leaf': 2},
 {'max_features': 'auto',
  'min_samples_leaf': 1},
 {'max_features': 0.1,
  'min_samples_leaf': 2},
 {'max_features': 0.2,
  'min_samples_leaf': 2},
 {'max_features': 0.05,
  'min_samples_leaf': 2},
 {'max_features': 0.15,
  'min_samples_leaf': 2},
 {'max_features': 0.2,
  'min_samples_leaf': 2},
 {'max_features': 0.2,
  'min_samples_leaf': 2},
 {'max_features': 0.2,
  'min_samples_leaf': 2},
 {'max_features': 0.15,
  'min_samples_leaf': 1},
 {'max_features': 0.2,
  'min_samples_leaf': 1},
 {'max_features': 0.2,
  'min_samples_leaf': 1},
 {'max_features': 0.2,
  'min_samples_leaf': 1},
 {'max_features': 0.2,
  'min_samples_leaf': 1},
 {'max_features': 0.2,
  'min_samples_leaf': 2},
 {'max_features': 'auto',
  'min_samples_leaf': 2},
 {'max_features': 0.2,
  'min_samples_leaf': 2},
 {'max_features': 0.2,
  'min_samples_leaf': 2},
 {'max_features': 0.1,
  'min_samples_leaf': 2},
 {'max_features': 0.2,
  'min_samples_leaf': 2},
 {'max_features': 0.55,
  'min_samples_leaf': 4}]

if __name__ == "__main__":
    args = opts().parse_args()
    Y_train = args.labels['labels']
    
    logging.info("Fitting classifiers")
    clfs = []
    Y_test = []
    for i, y in enumerate(Y_train.T):
        if args.col and i not in args.col:
            y_hat = y.mean()
            Y_test.append(y_hat*np.ones(args.test.shape[0]))
            clfs.append(y_hat)
            continue
            
        logging.info(i)
        kwargs = CFG[i]
        clf = RandomForestClassifier(
            n_estimators=128, criterion='entropy', max_depth=None,
            max_leaf_nodes=None, min_samples_split=1,
            n_jobs=-1, random_state=42, verbose=2,
            **kwargs)
        if len(np.unique(y)) == 1:
            Y_test.append(y[0]*np.ones(args.test.shape[0]))
            clfs.append(y[0])
        else:
            logging.info("Fitting")
            clf.fit(args.train, y)
            clfs.append(clf)
            logging.info("Predicting")
            p = clf.predict_proba(args.test)
            y = 1 - p[:,0]
            Y_test.append(y)
            
    logging.info("Saving predictions to %s" % args.output)
    test = load_npz('../../data/test.npz')
    Y_test = np.vstack(Y_test).T
    save_npz(args.output, ids=test['ids'], 
        header=args.labels['header'], labels=Y_test)
    del Y_test
        
    if args.classifiers:
        logging.info("Saving classifiers to %s" % args.classifiers)
        pickle.dump(clfs, args.classifiers, 2)
