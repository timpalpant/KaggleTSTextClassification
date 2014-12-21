#!/usr/bin/env python

'''
Fit weighted average of multiple predictions
'''

import sys, argparse
from common import *
from scipy.optimize import minimize

def fit_weights(Y, labels):
    w0 = np.ones(Y.shape[1]) / Y.shape[1]
    cons = ({'type': 'eq', 'fun': lambda x: x.sum()-1})
    b = [(0,1) for i in xrange(Y.shape[1])]
    f = lambda w: score_predictions(labels, (w*Y).sum(axis=1))
    res = minimize(f, w0)
    return res.x

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',
        help='Averaged predictions + weights (npz)')
    parser.add_argument('--pred', type=load_npz, action='append',
        required=True, help='Predicted labels (npz)')
    parser.add_argument('--labels', type=load_npz,
        help='Reference labels')
    parser.add_argument('--weights', type=load_npz,
        help='Reuse previously-fit weights')
    parser.add_argument('--unweighted', action='store_true',
        help='Unweighted average')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    print "Loading test data predictions"
    Y_i = np.array([p['labels'] for p in args.pred])
    nlabels = Y_i.shape[2]
    
    if args.unweighted:
        Y_avg = Y_i.mean(axis=0)
        weights = np.ones(Y_i.shape[0])
    elif args.weights:
        weights = args.weights['weights']
        Y_avg = np.zeros_like(Y_i[0])
        for i in xrange(Y_i.shape[0]):
            Y_avg += weights[:,i] * Y_i[i]
    else:
        print "Loading labels"
        labels = args.labels['labels']
    
        print "Averaging predictions"
        Y_avg = []
        weights = []
        for i in xrange(nlabels):
            Y = np.vstack([y[:,i] for y in Y_i]).T
            s = [score_predictions(labels[:,i], y[:,i]) for y in Y_i]
            print "Score for each prediction: %s" % s
            w = fit_weights(Y, labels[:,i])
            weights.append(w)
            print "Weights for label %d: %s" % (i, w)
            y_avg = (w*Y).sum(axis=1)
            s = score_predictions(labels[:,i], y_avg)
            print "Score for averaged predictions: %s" % s
            Y_avg.append(y_avg)
        weights = np.array(weights)
        Y_avg = np.vstack(Y_avg).T
    
    print "Writing output to %s" % args.output
    ids = args.pred[0]['ids']
    header = args.pred[0]['header']
    save_npz(args.output, weights=weights, ids=ids, 
             header=header, labels=Y_avg)
