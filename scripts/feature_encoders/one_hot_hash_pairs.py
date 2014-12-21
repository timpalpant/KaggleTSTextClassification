#!/usr/bin/env python

'''
Just the raw features
'''

import argparse, logging, gc
import numpy as np
from scipy import sparse
from common import load_npz, save_encoded_features
from sklearn.utils import murmurhash3_32
import mmh3
import pyhash

logging.basicConfig(level=logging.DEBUG)

def one_hot_hash(loader, output, d):
    logging.info("Loading raw data")
    bfs = loader['bfeatures']
    ffs = loader['ffeatures']
    ifs = loader['ifeatures']
    sfs = loader['sfeatures']
    npaircols = sfs.shape[1] + bfs.shape[1]
    X = np.hstack((sfs, bfs, ifs, ffs))
    del bfs, ffs, ifs, sfs
    
    nrows = X.shape[0]
    ncols = X.shape[1] + npaircols*(npaircols-1)/2
    ij = np.zeros((2, nrows*ncols), dtype=int) # row, col indices
    #hasher = pyhash.murmur3_32()
    for i, row in enumerate(X):
        if i % 50000 == 0: 
            logging.debug(i)
            gc.collect()
        start = i * ncols
        end = (i+1) * ncols
        ij[0,start:end] = i
        for j, x in enumerate(row):
            #ij[1,start+j] = abs(mmh3.hash(str((j,x)), 42)) % d
            ij[1,start+j] = murmurhash3_32(str((j,x)), seed=42, positive=True) % d
            #ij[1,start+j] = abs(hasher(str(j), str(x), seed=42)) % d
            #ij[1,start+j] = abs(hash((j,x))) % d
        j += start
        for j1 in xrange(npaircols):
            for j2 in xrange(j1):
                j += 1
                #ij[1,j] = abs(mmh3.hash(str((j1,row[j1],j2,row[j2])), 
                #                         seed=42)) % d
                ij[1,j] = murmurhash3_32(str((j1,row[j1],j2,row[j2])), 
                                         seed=42, positive=True) % d
                #ij[1,j] = abs(hasher(str(j1), str(row[j1]), str(j2), str(row[j2]), 
                #                     seed=42)) % d
                #ij[1,j] = abs(hash((j1,row[j1],j2,row[j2]))) % d
    data = np.ones(ij.shape[1]) # all ones
    X_hot = sparse.csr_matrix((data, ij), shape=(nrows, d))
    
    logging.info("Saving one-hotted data to output")
    save_encoded_features(output, X_hot)

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('features', type=load_npz,
        help='Features (npz)')
    parser.add_argument('output',
        help='Output features (npz)')
    parser.add_argument('-d', type=int, default=32,
        help='Dimension of hash table = 2**d')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    logging.info("One-hot encoding feature data")
    one_hot_hash(args.features, args.output, 2**args.d)
