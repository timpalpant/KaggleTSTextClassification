#!/usr/bin/env python

'''
Output overrides from matched entries in test and train data
'''

import argparse, logging
import numpy as np
from common import load_npz, save_npz, score_predictions

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
    Y_train = []
    Y_train_override = []
    logging.info("Getting mean labels for matched ids")
    for line in args.matches:
        test_id, train_ids = line.rstrip().split('\t', 1)
        ids.append(int(test_id))
        train_ids = [rlookup[int(id)] for id in train_ids.split(',')]
        y_hat = labels[train_ids].mean(axis=0)
        Y.append(y_hat)
        Y_train.append(labels[train_ids])
        Y_train_override.append(np.tile(y_hat, (len(train_ids), 1)))
        
    ids = np.asarray(ids)
    Y = np.asarray(Y)
    Y_train = np.vstack(Y_train)
    Y_train_override = np.vstack(Y_train_override)
    logging.info("Override cost = %f" \
        % score_predictions(Y_train, Y_train_override))
    
    logging.info("Saving overridden predictions to %s" % args.output)
    save_npz(args.output, ids=ids, labels=Y,
        header=args.labels['header'])
