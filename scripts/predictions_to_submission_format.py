#!/usr/bin/env python

'''
Write predictions in submission format
'''

import sys, argparse
from common import *

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pred', type=load_npz,
        help='Predicted labels (npz)')
    parser.add_argument('output', type=argparse.FileType('w'),
        help='File for submission (csv)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    save_predictions(args.pred['ids'], args.pred['header'], 
        args.pred['labels'], args.output)