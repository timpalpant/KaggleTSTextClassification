#!/usr/bin/env python

'''
Find exact matches of the test features in the train data
'''

import argparse
from collections import defaultdict

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=argparse.FileType('r'),
        help='Training features (csv)')
    parser.add_argument('test', type=argparse.FileType('r'),
        help='Test features (csv)')
    parser.add_argument('output', type=argparse.FileType('w'),
        help='Output matched ids (tsv)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    
    # Load training data into dict
    print "Making lookup table from train data"
    header = args.train.readline()
    vlookup = defaultdict(list)
    for i, line in enumerate(args.train):
        if i % 100000 == 0: print i
        id, data = line.rstrip().split(',', 1)
        vlookup[data].append(id)
    print "Lookup table has %d unique entries" % len(vlookup)
    print "Processed %d lines" % i
        
    header = args.test.readline()
    nmatches = 0
    print "Matching lines in test data"
    for i, line in enumerate(args.test):
        if i % 100000 == 0: print i
        id, data = line.rstrip().split(',', 1)
        if data in vlookup:
            nmatches += 1
            print >>args.output, '%s\t%s' % (id, ','.join(vlookup[data]))
    print "Matched %d / %d lines in test data" % (nmatches, i)