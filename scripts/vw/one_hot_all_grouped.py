#!/usr/bin/env python

'''
Encode features for VW

Groups a, b, c, d are the most important features
Subsequent groups are less important
'''

import argparse
from itertools import izip
import numpy as np
from common import load_npz
import gzip

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=load_npz,
        help='Train features (npz)')
    parser.add_argument('output',
        help='Output train features (vw)')
    return parser

def main(args):
    print "Loading raw data"
    features = {'b': args.train['bfeatures'],
                'f': args.train['ffeatures'],
                'i': args.train['ifeatures'],
                's': args.train['sfeatures']}
    ids = args.train['ids']
    bfeatures = features['b']
    ffeatures = features['f']
    ifeatures = features['i']
    sfeatures = features['s']
    
    # best boolean features, from an analysis of MI vs. labels
    #best_bs = [41, 43, 46, 48] # included in best_features
    # clusters of ifeatures, from analysis of ifeature entropy
    i_groups = list(reversed([range(0,6), range(6,12), range(12,18),
                              range(18,24), range(24, 30)]))
    # most important features, from RF
    best_features = {'b': (41, 43, 46, 48, 11, 42, 44, 45, 47, 49),
                     'f': (4, 5, 6, 9, 10, 11, 15, 16, 17, 20, 21, 26, 27, 28, 31, 
                           32, 33, 37, 38, 39, 42, 43, 48, 49, 50, 51, 53),
                     's': (4, 7)}
    lesser_features = {k: tuple(set(xrange(features[k].shape[1])) - set(v))
                       for k, v in best_features.iteritems()}
    
    print "Writing in VW format"
    with gzip.GzipFile(args.output, 'w') as out:
        for id, bfs, ffs, ifs, sfs in izip(ids, bfeatures, ffeatures, ifeatures, sfeatures):
            if id % 10000 == 0: print id
            out_row = ["'%d" % id]
            n = 96 # ASCII character 'a'
                
            x = {'b': bfs, 'f': ffs, 'i': ifs, 's': sfs}
            for k in ('f', 's', 'b'):
                n += 1
                out_row += ['|%s' % chr(n)]
                out_row += ['%s%d_%s' % (k, col, x[k][col]) 
                            for col in best_features[k]]
                
            for ig in i_groups:
                n += 1
                out_row += ['|%s' % chr(n)] 
                out_row += ['i%d_%s' % (i,ifs[i]) for i in ig]
                
            for k in ('f', 's', 'b'):
                n += 1
                out_row += ['|%s' % chr(n)]
                out_row += ['%s%d_%s' % (k, col, x[k][col]) 
                            for col in lesser_features[k]]
                
            print >>out, ' '.join(out_row)

if __name__ == "__main__":
    args = opts().parse_args()
    main(args)
        