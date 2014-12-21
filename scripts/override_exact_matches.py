#!/usr/bin/env python

'''
Override predictions with exact matches from training data
'''

import argparse, logging
from common import load_npz, save_npz

logging.basicConfig(level=logging.DEBUG)

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=load_npz,
        help='Predicted labels (npz)')
    parser.add_argument('overrides', type=load_npz,
        help='Overrides labels (npz)')
    parser.add_argument('output',
        help='Output label predictions (npz)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    
    Y = args.predictions['labels']
    label_ids = args.predictions['ids']
    rlookup = {id: row for row, id in enumerate(label_ids)}
    override_rows = [rlookup[id] for id in args.overrides['ids']]
    Y[override_rows] = args.overrides['labels']
    
    logging.info("Saving overridden predictions to %s" % args.output)
    save_npz(args.output, ids=args.predictions['ids'], 
        header=args.predictions['header'], labels=Y)