#!/usr/bin/env python

'''
Encode features for VW
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
    ids = args.train['ids']
    bfeatures = args.train['bfeatures']
    ffeatures = args.train['ffeatures']
    ifeatures = args.train['ifeatures']
    sfeatures = args.train['sfeatures']
    print "Writing in VW format"
    with gzip.GzipFile(args.output, 'w') as out:
        for id, bfs, ffs, ifs, sfs in izip(ids, bfeatures, ffeatures, ifeatures, sfeatures):
            if id % 10000 == 0: print id
            b = ' '.join('b%d_%s' % (i,x) for i,x in enumerate(bfs))
            f = ' '.join('f%d_%s' % (i,x) for i,x in enumerate(ffs))
            i = ' '.join('i%d_%s' % (i,x) for i,x in enumerate(ifs))
            s = ' '.join('s%d_%s' % (i,x) for i,x in enumerate(sfs))
            print >>out, "'%d |b %s |f %s |i %s |s %s" % (id, b, f, i, s)

if __name__ == "__main__":
    args = opts().parse_args()
    main(args)
        