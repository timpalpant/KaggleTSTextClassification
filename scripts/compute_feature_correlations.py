#!/usr/bin/env python

'''
Compute correlations between labels
'''

import argparse
import numpy as np
from common import *
from information import entropy, mutual

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('features', type=load_npz,
        help='Training data features (npz)')
    parser.add_argument('output',
        help='Output data file (npz)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    cfs = args.features['ffeatures'] # continuous features
    bfs = args.features['bfeatures']
    ifs = args.features['ifeatures']
    sfs = args.features['sfeatures']
    dfs = np.hstack((bfs,ifs,sfs,cfs)) # discrete features
    n = dfs.shape[1]
    print "%d features" % n

    I = np.zeros((n,n)) # mutual information between features i, j
    H = np.zeros((n,n)) # entropy of joint distribution of features i, j

    # MI of discrete vs. discrete
    for i in xrange(dfs.shape[1]):
        print i
        for j in xrange(i):
            print '...%s' % j
            H[i,j] = entropy.discrete(dfs[:,[i,j]])
            I[i,j] = mutual.discrete(dfs[:,i], dfs[:,j])

    # MI of continuous vs. continuous
    #offset = dfs.shape[1]
    #for i in xrange(cfs.shape[1]):
    #    print i
    #    for j in xrange(i):
    #        print j
    #        H[offset+i,offset+j] = entropy.continuous(cfs[:,[i,j]])
    #        I[offset+i,offset+j] = mutual.continuous(cfs[:,[i,j]])

    # MI of continuous vs. discrete
    
    I += I.T
    H += H.T
    
    save_npz(args.output, i=I, h=H)
