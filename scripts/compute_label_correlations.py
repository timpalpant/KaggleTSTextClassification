#!/usr/bin/env python

'''
Compute correlations between labels
'''

import sys, argparse
from common import *
from information import mutual, entropy

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('labels',
        help='Training data labels (CSV)')
    parser.add_argument('output',
        help='Output data file (Pickle)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    print "Loading labels from %s" % args.labels
    _, _, labels = load_labels(args.labels)
    
    print "Computing means"
    means = labels.mean(axis=0)
    nlabels = len(means)
    
    print "Computing correlations"
    C = np.eye(nlabels)
    for i in xrange(nlabels):
        print i
        for j in xrange(i):
            C[i,j] = np.corrcoef(labels[:,i], labels[:,j])[0,1]
    C += C.T # symmetrize
    C[np.diag_indices_from(C)] /= 2
    
    print "Computing MI"
    mi = np.zeros((nlabels, nlabels))
    for i in xrange(nlabels):
        print i
        mi[i,i] = entropy.discrete(labels[:,i])
        for j in xrange(i):
            try: mi[i,j] = mutual.discrete(labels[:,i], labels[:,j])
            except Exception, e: print >>sys.stderr, e
    mi += mi.T # symmetrize
    mi[np.diag_indices_from(mi)] /= 2
    
    print "Saving results to %s" % args.output
    results = {'means': means,
               'C': C,
               'MI': mi}
    with open(args.output, 'w') as fd:
        pickle.dump(results, fd, -1)