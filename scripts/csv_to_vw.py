#!/usr/bin/env python

'''
Convert data to vw file
'''

import argparse
from common import *

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train',
        help='Training data (csv)')
    parser.add_argument('train_labels',
        help='Training data labels (csv)')
    parser.add_argument('test',
        help='Test data (csv)')
    parser.add_argument('train_out',
        help='Training data output file (npz)')
    parser.add_argument('train_labels_out',
        help='Training data labels output file (npz)')
    parser.add_argument('test_out',
        help='Test data output file (npz)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    for input, output in ((args.train, args.train_out),
                          (args.train_labels, args.train_labels_out),
                          (args.test, args.test_out)):
        loader = guess_loader(input)
        print >>sys.stderr, "Loading data from %s" % input
        data = loader(input)
        print >>sys.stderr, "Saving to %s" % output
        save_npz(output, **data)
        del data