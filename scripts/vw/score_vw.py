#!/usr/bin/env python

'''
Score VW predictions on validation set
'''

import argparse, gzip
from itertools import izip
import numpy as np
from common import load_npz, score_predictions

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) 

def load_vw_predictions(fd):
    y = np.array([float(line.rstrip().split()[0]) 
                  for line in fd])
    p = sigmoid(y)
    return p

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pred', type=argparse.FileType('r'),
        help='VW predictions')
    parser.add_argument('labels', type=load_npz,
        help='Train labels (npz)')
    parser.add_argument('--label', type=int, required=True,
        help='Label to score against')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    print "Loading VW predictions"
    p = load_vw_predictions(args.pred)
    print "Loading labels"
    y = args.labels['labels'][:,args.label]
    s = score_predictions(y, p)
    print "Score: %s" % s
    
        
