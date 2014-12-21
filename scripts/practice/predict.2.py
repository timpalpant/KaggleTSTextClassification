#!/usr/bin/env python

'''
Make predictions for the test data

2. Use a KNN classifier
'''

import argparse
from common import *
from scipy.stats import itemfreq
from sklearn.neighbors import DistanceMetric

def prepare_features(data):
    ifeatures = np.asarray(data['ifeatures'], dtype=float)
    ifeatures[ifeatures==-1] = -100.0
    sfeatures = np.asarray(data['sfeatures'], dtype=float)
    # Map categorical data (hashes) to frequencies
    h = {k: v for k, v in itemfreq(sfeatures)}
    for i in xrange(sfeatures.shape[0]):
        for j in xrange(sfeatures.shape[1]):
            if sfeatures[i,j] == -1:
                sfeatures[i,j] = -100.0
            else:
                sfeatures[i,j] = h[sfeatures[i,j]]
    bfeatures = np.asarray(data['bfeatures'], dtype=float)
    bfeatures[bfeatures==-1] = -100.0
    X = np.hstack((bfeatures, data['ffeatures'],
                   ifeatures, sfeatures))
    return X
    
def d(x1, x2):
    

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
    X = prepare_features(args.train)
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    Y = args.labels['labels']
    
    print "Training classifier"
    clf = neighbors.KNeighborsClassifier(n_neighbors=1)
    clf.fit(X, Y)
    del X, Y
    
    print "Predicting"
    X = scaler.transform(prepare_features(args.test))
    Y = clf.predict(X)
    
    print "Saving predictions"
    save_npz(args.output, ids=args.test['ids'], 
        header=args.labels['header'], labels=Y)
    