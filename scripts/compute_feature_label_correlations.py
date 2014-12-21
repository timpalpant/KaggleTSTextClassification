#!/usr/bin/env python

'''
Compute mutual information between individual features
and labels
'''

import argparse
from common import *
from information import mutual, entropy

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('features', type=load_npz,
        help='Training data features (npz)')
    parser.add_argument('labels', type=load_npz,
        help='Training data labels (npz)')
    parser.add_argument('output',
        help='Output data file (npz)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    print "Loading labels"
    Y = args.labels['labels']
    print "Computing entropy of each label"
    p = np.mean(Y, axis=0)
    h = -(p*np.log(p) + (1-p)*np.log(1-p))
    h[np.logical_or(p==0,p==1)] = 0 # 0*log(0) := 0
    results = {'p': p, 'h': h}
    
    print "Computing correlations and MI"
    for t in ('bfeatures', 'ifeatures', 'sfeatures'):
        print t
        print '...loading'
        X = args.features[t]
        print '...H'
        h = np.asarray([entropy.discrete(x) for x in X.T])
        print '...I'
        h_y0 = np.asarray([[entropy.discrete(x[np.logical_not(y)]) for y in Y.T]
                           for x in X.T])
        h_y1 = np.asarray([[entropy.discrete(x[y]) for y in Y.T]
                           for x in X.T])
        h_y = p*h_y1 + (1-p)*h_y0
        mi = np.tile(h, (len(Y.T),1)).T - h_y
        #mi = np.asarray([[mutual.discrete(x, y) for y in Y.T]
        #                 for x in X.T])
        results[t+'_h'] = h
        results[t+'_mi'] = mi
    
    print "ffeatures"
    print "...loading"
    X = args.features['ffeatures']
    print '...H'
    h = np.asarray([entropy.continuous(x) for x in X.T])
    print '...I'
    h_y0 = np.asarray([[entropy.continuous(x[np.logical_not(y)]) for y in Y.T]
                       for x in X.T])
    h_y1 = np.asarray([[entropy.continuous(x[y]) for y in Y.T]
                       for x in X.T])
    h_y = p*h_y1 + (1-p)*h_y0
    mi = np.tile(h, (len(Y.T),1)).T - h_y
    results['ffeatures_h'] = h
    results['ffeatures_mi'] = mi
    
    print "Saving results to %s" % args.output
    save_npz(args.output, **results)