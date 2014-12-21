#!/usr/bin/env python

'''
Score VW predictions on validation set
'''

import argparse, gzip, subprocess, shlex
from itertools import izip, product
import numpy as np
from common import load_npz, score_predictions

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) 

def load_vw_predictions(fd):
    y = np.array([float(line.rstrip().split()[0]) 
                  for line in fd])
    p = sigmoid(y)
    return p
    
def run_vw(label, learning_rate=0.5, learning_rate2=0.5, l1=0, b=28):
    # Train with L1 regularization (LASSO)
    cmd = 'vw -d ../../data/train.train.one_hot_all_grouped.y%d.vw -f model_y%d.grouped.l1.vw --loss_function logistic -b %d -l %f -q bb -q jj -q bj -q bc -q ab -q dj -q db -q cc -q dd --passes 3 --hash all --random_seed 42 --compressed -c --l1 %s' % (label, label, b, learning_rate, l1)
    subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT)
    # Re-train with selected features
    cmd = 'vw -d ../../data/train.train.one_hot_all_grouped.y%d.vw -f model_y%d.grouped.vw --loss_function logistic -b %d -l %f -q bb -q jj -q bj -q bc -q ab -q dj -q db -q cc -q dd --passes 3 --hash all --random_seed 42 --compressed -c --feature_mask model_y%d.grouped.l1.vw' % (label, label, b, learning_rate2, label)
    subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT)
    # predict heldout data
    cmd = 'vw -d ../../data/train.validate.one_hot_all_grouped.vw -i model_y%d.grouped.vw -p preds_y%d.grouped.p.txt --compressed -c' % (label, label)
    subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT)
    with open('preds_y%d.grouped.p.txt' % label) as fd:
        p = load_vw_predictions(fd)
    return p

if __name__ == "__main__":
    labels = load_npz('../../data/trainLabels.validate.npz')['labels']
    
    for i in (32,11,5,8,6,28,9,30,31):
        print "Tuning label %d" % i
        y = labels[:,i]
    
        best = None
        min_score = np.inf
        cfg = product((0.1, 0.2, 0.3), (0.1, 0.2, 0.3), ('1e-8', '5e-8'))
        for lr, lr2, l1 in cfg:
            print "lr = %f, lr2 = %f, l1 = %s" % (lr, lr2, l1)
            p = run_vw(i, lr, lr2, l1)
            s = score_predictions(y, p)
            print "Score: %s" % s
            if s < min_score:
                best = [lr, l1]
                min_score = s
            
        print "Best: %s (with score = %f)" % (best, min_score)
    
        
