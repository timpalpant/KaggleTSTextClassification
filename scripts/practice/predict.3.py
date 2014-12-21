#!/usr/bin/env python

'''
Make predictions for the test data

3. Use naive Bayes, on just the boolean features,
   with a separate classifier for each label.
'''

import argparse
from common import *
from scipy.stats import itemfreq
from sklearn.naive_bayes import GaussianNB

def prepare_features(data):
    return data['bfeatures'], data['ffeatures']

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=load_npz,
        help='Training features (npz)')
    parser.add_argument('labels', type=load_npz,
        help='Training labels (npz)')
    parser.add_argument('test', type=load_npz,
        help='Test features (npz)')
    parser.add_argument('output',
        help='Output label predictions (npz)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    
    print "Loading and preparing data"
    X1, X2 = prepare_features(args.train)
    Y = args.labels['labels']
    
    print "Training classifiers"
    clf1s = []
    for i in xrange(Y.shape[1]):
        print i
        clf = naive_bayes.BernoulliNB(fit_prior=True)
        clf.fit(X1, Y[:,i])
        clf1s.append(clf)
    del X1
    
    clf2s = []
    for i in xrange(Y.shape[1]):
        print i
        clf = naive_bayes.GaussianNB()
        clf.fit(X2, Y[:,i])
        clf2s.append(clf)
    del X2, Y
    
    print "Predicting"
    X1, X2 = prepare_features(args.test)
    p1 = np.vstack([clf.predict_proba(X1)[:,0] for clf in clf1s])
    Y1 = 1 - p1.T
    p2 = np.vstack([clf.predict_proba(X2)[:,0] for clf in clf2s])
    Y2 = 1 - p2.T
    Y = (Y1 + Y2) / 2
    Y[:,13] = 0
    Y[np.isnan(Y)] = 0
    
    print "Saving predictions"
    save_npz(args.output, ids=args.test['ids'], 
        header=args.labels['header'], labels=Y)
    