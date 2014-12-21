#!/usr/bin/env python

'''
Make predictions for the test data

1. Use a random forest, as described:
https://www.kaggle.com/c/tradeshift-text-classification/forums/t/10518/random-forest-benchmark
'''

import argparse
from common import *
from scipy.stats import itemfreq

def prepare_features(data):
    ifeatures = np.asarray(data['ifeatures'], dtype=float)
    #ifeatures[ifeatures==-1] = -999.0
    sfeatures = np.asarray(data['sfeatures'], dtype=float)
    # Map categorical data (hashes) to frequencies
    #h = {k: v for k, v in itemfreq(sfeatures)}
    #for i in xrange(sfeatures.shape[0]):
    #    for j in xrange(sfeatures.shape[1]):
    #        if sfeatures[i,j] == -1:
    #            sfeatures[i,j] = -999.0
    #        else:
    #            sfeatures[i,j] = h[sfeatures[i,j]]
    bfeatures = np.asarray(data['bfeatures'], dtype=float)
    #bfeatures[bfeatures==-1] = -999.0
    X = np.hstack((bfeatures, data['ffeatures'],
                   ifeatures, sfeatures))
    return X

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

def cross_validate_classifier(clf, features, labels, cv=5):
    kf = cross_validation.KFold(len(features), n_folds=cv)
    scores = []
    for train, test in kf:
        clf.fit(features[train], labels[train])
        Y = np.vstack([v[:,0] for v in clf.predict_proba(features[test])])
        pred = 1 - Y.T
        score = score_predictions(labels[test], pred)
    return np.asarray(scores)

if __name__ == "__main__":
    args = opts().parse_args()
    
    print "Loading and preparing data"
    X = prepare_features(args.train)
    Y = args.labels['labels']
    
    print "Cross-validating classifier"
    clf = ensemble.RandomForestClassifier(
        n_estimators=48, min_samples_split=1, max_depth=None,
        criterion='entropy',
        n_jobs=-1, random_state=42, verbose=2)
    scores = cross_validate_classifier(clf, X, Y)
    print "Classifier performance = %f (+/-) %f" \
        % (scores.mean(), scores.std())
        
    print "Training classifier on full dataset"
    clf.fit(X, Y)
    del X, Y
    
    print "Predicting labels for test set"
    X = prepare_features(args.test)
    p = np.vstack([v[:,0] for v in clf.predict_proba(X)])
    Y = 1 - p.T
    
    print "Saving predictions"
    save_npz(args.output, ids=args.test['ids'], 
        header=args.labels['header'], labels=Y,
        feature_importances=clf.feature_importances_)
    
