#!/usr/bin/env python

'''
Write predictions in submission format
'''

import sys, argparse, gzip
from common import *

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=gzip.GzipFile,
        help='File for submission (csv)')
    parser.add_argument('output',
        help='Predicted labels (npz)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    print "Loading predictions"
    pred = load_predictions(args.csv)
    print "Saving predictions to %s" % args.output
    save_npz(args.output, **pred)