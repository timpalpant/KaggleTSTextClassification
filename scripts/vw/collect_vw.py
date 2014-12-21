#!/usr/bin/env python

'''
Collect VW predictions into npz file
'''

import argparse, glob
import numpy as np
from common import load_npz, save_npz

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) 

def load_vw_predictions(fd):
    ids = []
    y = []
    for line in fd:
        entry = line.rstrip().split()
        id = int(entry[1])
        ids.append(id)
        y_i = float(entry[0])
        y.append(y_i)
    ids = np.array(ids)
    y = np.array(y)
    p = sigmoid(y)
    return ids, p

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pred', type=glob.glob,
        help='VW predictions pattern')
    parser.add_argument('output',
        help='Output file (npz)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    print "Loading VW predictions"
    ids = load_vw_predictions(open(args.pred[0]))[0]
    print "Loading predictions for %d ids" % len(ids)
    Y = np.zeros((len(ids), 33))
    for fn in args.pred:
        i = int(fn.split('.')[0].split('_')[-1][1:])
        print "Label %d" % i
        Y[:,i] = load_vw_predictions(open(fn))[1]
    labels = load_npz('../../data/trainLabels.npz')
    header = labels['header']
    save_npz(args.output, header=header, ids=ids, labels=Y)
    
        
