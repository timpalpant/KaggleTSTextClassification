#!/usr/bin/env python

'''
Make predictions for the test data

0. Just guess the mean frequency of each label
'''

import argparse
from common import *

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
    Y = args.labels['labels']
    Ym = np.mean(Y, axis=0)
    nsamples = len(args.test['bfeatures'])
    Y = np.tile(Ym, (nsamples,1))
    
    print "Saving predictions"
    save_npz(args.output, ids=args.test['ids'], 
        header=args.labels['header'], labels=Y)
    