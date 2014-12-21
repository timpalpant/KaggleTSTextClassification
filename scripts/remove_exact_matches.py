#!/usr/bin/env python

'''
Remove exact matches from the train/test data.
We are going to override them with the values from the training data,
so there's no need to fit the model to them. We want to fit
the model to th
'''

import argparse, logging
import numpy as np
from common import load_npz, save_npz

logging.basicConfig(level=logging.DEBUG)

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('labels', type=load_npz,
        help='Training labels (csv)')
    parser.add_argument('matches', type=argparse.FileType('r'),
        help='Matched test/train ids (tsv)')
    parser.add_argument('output',
        help='Output overrides (npz)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    
    logging.info("Making lookup table for row ids")
    labels = args.labels['labels']
    rlookup = {id: row for row, id in enumerate(args.labels['ids'])}
    
    ids = []
    Y = []
    logging.info("Getting mean labels for matched ids")
    for line in args.matches:
        test_id, train_ids = line.rstrip().split('\t', 1)
        ids.append(int(test_id))
        train_ids = [rlookup[int(id)] for id in train_ids.split(',')]
        y_hat = labels[train_ids].mean(axis=0)
        Y.append(y_hat)
        
    ids = np.asarray(ids)
    Y = np.asarray(Y)
    
    logging.info("Saving overridden predictions to %s" % args.output)
    save_npz(args.output, ids=ids, labels=Y,
        header=args.labels['header'])