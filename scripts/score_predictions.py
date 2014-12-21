#!/usr/bin/env python

'''
Compute correlations between labels
'''

import sys, argparse
from common import *

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ref', type=load_npz,
        help='True labels (npz)')
    parser.add_argument('pred', type=load_npz,
        help='Predicted labels (csv)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    scores = score_predictions(args.ref['labels'], args.pred['labels'], axis=0)
    for s, id in reversed(sorted((s,id) for id, s in enumerate(scores))):
        print "Label %d: %s" % (id, s)
    print "Average log-loss: %s" % scores.mean()
