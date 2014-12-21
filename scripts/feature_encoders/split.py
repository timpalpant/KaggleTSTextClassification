#!/usr/bin/env python

'''
Just the raw features
'''

import argparse, logging, os
import numpy as np
from common import load_npz, save_npz

logging.basicConfig(level=logging.DEBUG)

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('features', type=load_npz,
        help='Features (npz)')
    parser.add_argument('output',
        help='Split features (npz)')
    parser.add_argument('-n', type=int, required=True,
        help='Number of files to split into')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    basename, ext = os.path.splitext(args.output)
    pattern = basename + '.%d' + ext
    logging.info("Loading data")
    data = dict(args.features)
    nrows = data['ids'].shape[0]
    nrows_per_file = nrows / args.n
    logging.info("Writing splits to output")
    for i in xrange(args.n):
        logging.info(i)
        split = {k: v[i*nrows_per_file:(i+1)*nrows_per_file]
                 for k, v in data.iteritems()}
        save_npz(pattern % i, **split)   