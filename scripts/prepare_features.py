'''
Prepare features from "raw" data for predictors
(note the "raw" data must first be converted to npz
using csv_to_npz.py)

@author Timothy Palpant <tim@palpant.us>
@date October 18, 2014
'''

import os, logging, gc
import cPickle as pickle
import numpy as np
from scipy import sparse
from sklearn.utils import murmurhash3_32

class EncodingCache(object):
    '''Cache the encoding materializations we have generated before'''
    cachedir = '/Users/timpalpant/Documents/Workspace/kaggle/TextClassification/data/materializations/'
    enabled = True
    
    @classmethod
    def get(cls, encoder, data, indices):
        # Check if there is a saved copy of this encoding
        if cls.contains(encoder, data, indices):
            return cls.cache_get(encoder, data, indices)
        enc = encoder()
        X = enc.prepare(data, indices)
        if cls.enabled:
            cls.put(encoder, data, indices, enc, X)
        return enc, X
        
    @classmethod
    def cache_get(cls, encoder, data, indices):
        logging.info("Loading encoded materialization from cache")
        pkl, npz = cls.hash(encoder, data, indices)
        with open(pkl, 'r') as fd:
            enc = pickle.load(fd)
        loader = np.load(npz)
        try: # reconstruct sparse arrays
            X = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                                  shape=loader['shape'])
        except: # dense arrays
            X = loader['X']
        return enc, X
        
    @classmethod
    def contains(cls, encoder, data, indices):
        pkl, npz = cls.hash(encoder, data, indices)
        return os.path.isfile(pkl)
        
    @classmethod
    def put(cls, encoder, data, indices, enc, X):
        logging.info("Saving encoded materialization to cache")
        pkl, npz = cls.hash(encoder, data, indices)
        with open(pkl, 'w') as fd:
            pickle.dump(enc, fd, pickle.HIGHEST_PROTOCOL)
        try: # sparse arrays
            np.savez(npz, data=X.data, indices=X.indices, 
                     indptr=X.indptr, shape=X.shape)
        except: # dense arrays
            np.savez(npz, X=X)
            
    @classmethod
    def hash(cls, encoder, data, indices):
        npzname = os.path.abspath(data.fid.name)
        ih = None
        if indices is not None:
            tmp = indices.flags.writeable
            indices.flags.writeable = False
            ih = hash(indices.data)
            indices.flags.writeable = tmp
        h = str(abs(hash((str(encoder), npzname, ih))))
        pkl = cls.cachedir+h+'.pkl'
        npz = cls.cachedir+h+'.npz'
        return pkl, npz

class TSFeatureEncoder(object):
    '''
    Takes the "raw" features from npz files
    and performs various encoding / engineering operations.
    
    If indices are provided, they should represent a subset
    of the rows in the feature matrix (i.e. for cross-validation)
    '''
    pass
    
class TSRawEncoder(TSFeatureEncoder):
    '''Just return all of the "raw" features, no processing'''
    def prepare(self, features, indices=None):
        logging.info("Preparing raw feature matrix")
        bfs = features['bfeatures']
        ffs = features['ffeatures']
        ifs = features['ifeatures']
        sfs = features['sfeatures']
        if indices is not None:
            bfs = bfs[indices]
            ffs = ffs[indices]
            ifs = ifs[indices]
            sfs = sfs[indices]
        X = np.hstack((bfs, ffs, ifs, sfs))
        del bfs, ffs, ifs, sfs
        return X

class TSOneHotAllEncoder(TSFeatureEncoder):
    '''one-hot encode everything exceeding frequency cutoff'''
    freq_cutoff = 5
    float_decimals = 2
    
    def __init__(self):
        # The first time prepare is called, it will make a new encoder.
        # Subsequent calls will re-use this encoder.
        self.encoder = None
    
    def prepare(self, features, indices=None, dtype=float):
        logging.info("One-hot encoding all features")
        bfs = features['bfeatures']
        ffs = features['ffeatures']
        ifs = features['ifeatures']
        sfs = features['sfeatures']
        if indices is not None:
            bfs = bfs[indices]
            ffs = ffs[indices]
            ifs = ifs[indices]
            sfs = sfs[indices]
        X = np.hstack((bfs, ffs, ifs, sfs))
        del bfs, ffs, ifs, sfs
        if self.encoder is None:
            self.encoder = OneHotEncoder()
            self.encoder.fit(X, self.freq_cutoff)
        X = self.encoder.transform(X, dtype)
        return X

class TSOneHotHashingEncoder(TSFeatureEncoder):
    '''one-hot encode everything with hashing trick'''
    D = 2 ** 20
    float_decimals = 2
    
    def prepare(self, features, indices=None, dtype=float):
        logging.info("One-hot hashing all features")
        bfs = features['bfeatures']
        ffs = features['ffeatures']
        ifs = features['ifeatures']
        sfs = features['sfeatures']
        if indices is not None:
            bfs = bfs[indices]
            ffs = ffs[indices]
            ifs = ifs[indices]
            sfs = sfs[indices]
        X = np.hstack((bfs, ffs, ifs, sfs))
        del bfs, ffs, ifs, sfs
        nrows = X.shape[0]
        ncols = X.shape[1]
        ij = np.zeros((2, nrows*ncols), dtype=int) # row, col indices
        for i, row in enumerate(X):
            if i % 100000 == 0: logging.debug(i)
            start = i * ncols
            end = (i+1) * ncols
            ij[0,start:end] = i
            for j, x in enumerate(row):
                ij[1,start+j] = murmurhash3_32('%d_%s' % (j,x), seed=42, positive=True) % self.D
        data = np.ones(ij.shape[1], dtype=dtype) # all ones
        X = sparse.csr_matrix((data, ij), shape=(nrows, self.D), dtype=dtype) 
        return X

class TSOneHotHashingStringPairsEncoder(TSOneHotHashingEncoder):
    '''one-hot encode everything with hashing trick, plus pairs of string features'''
    def prepare(self, features, indices=None, dtype=float):
        X1 = super(TSOneHotHashingStringPairsEncoder, self).prepare(features, indices)
        logging.info("One-hot hashing pairs of string features")
        sfs = features['sfeatures']
        if indices is not None:
            sfs = sfs[indices]
        nrows = sfs.shape[0]
        ncols = sfs.shape[1]*(sfs.shape[1]-1) / 2
        ij = np.zeros((2, nrows*ncols), dtype=int) # row, col indices
        for i, row in enumerate(sfs):
            if i % 100000 == 0: logging.debug(i)
            start = i * ncols
            end = (i+1) * ncols
            ij[0,start:end] = i
            ij[1,start:end] = [murmurhash3_32('%d_%s_x_%d_%s' % (j1,x1,j2,row[j2]), 
                                              seed=42, positive=True) % self.D
                               for j1, x1 in enumerate(row)
                               for j2 in xrange(j1)]
        data = np.ones(ij.shape[1], dtype=dtype) # all ones
        X2 = sparse.csr_matrix((data, ij), shape=(nrows, self.D), dtype=dtype) 
        X = X1 + X2
        X.data[X.data > 1] = 1
        return X

class TSOneHotHashingPairsEncoder(TSOneHotHashingEncoder):
    '''
    one-hot encode everything with hashing trick, 
    plus pairs of string and boolean features
    '''
    def prepare(self, features, indices=None, dtype=float):
        X1 = super(TSOneHotHashingPairsEncoder, self).prepare(features, indices)
        logging.info("One-hot hashing pairs of string and boolean features")
        sfs = features['sfeatures']
        bfs = features['bfeatures']
        if indices is not None:
            sfs = sfs[indices]
            bfs = bfs[indices]
        X = np.hstack((sfs, bfs))
        del sfs, bfs
        nrows = X.shape[0]
        ncols = X.shape[1]*(X.shape[1]-1) / 2
        ij = np.zeros((2, nrows*ncols), dtype=int) # row, col indices
        for i, row in enumerate(X):
            if i % 100000 == 0: logging.debug(i)
            start = i * ncols
            end = (i+1) * ncols
            ij[0,start:end] = i
            ij[1,start:end] = [murmurhash3_32('%d_%s_x_%d_%s' % (j1,x1,j2,row[j2]), 
                                              seed=42, positive=True) % self.D
                               for j1, x1 in enumerate(row)
                               for j2 in xrange(j1)]
        data = np.ones(ij.shape[1], dtype=dtype) # all ones
        X2 = sparse.csr_matrix((data, ij), shape=(nrows, self.D), dtype=dtype) 
        X = X1 + X2
        X.data[X.data > 1] = 1
        return X

class OneHotEncoder(object):
    '''
    will transform categorical feature X into one-hot encoded features X_hot
    
    I tried to use sklearn's OneHotEncoder, but it doesn't offer a 
    good way to reapply the same encoding to the test data if the
    test data contains new (never seen) values. There's an open ticket
    to address this.
    '''
    def fit(self, X, freq_cutoff=0):
        '''
        Fit encoder to values in @X having frequency > @freq_cutoff
        '''
        logging.debug("Making one-hot encoder for %dx%d feature matrix" % X.shape)
        self.value_to_col = []
        offset = 0
        for i, x in enumerate(X.T):
            logging.debug("processing column %d" % i)
            values = self.unique_values(x, freq_cutoff)
            d = {v: j+offset for j, v in enumerate(values)}
            offset += len(d)
            self.value_to_col.append(d)
        self.ncols = offset + len(self.value_to_col)
        
    def transform(self, X, dtype=np.float64):
        '''
        Apply encoder to values in @X.
        Returns a sparse boolean matrix.
        '''
        # Make a sparse boolean matrix with one-hot encoded features
        # one column for each categorical value of each colum in @X
        # plus one column to signify 'other' for each column of @X
        nrows = X.shape[0]
        ncols = X.shape[1]
        logging.debug("Making %dx%d one-hot matrix" % (nrows, self.ncols))  
        i = np.zeros(nrows*ncols, dtype=np.uint32)
        j = np.zeros(nrows*ncols, dtype=np.uint32)
        data = np.ones(i.shape[0], dtype=np.uint8) # all ones
        for k in xrange(nrows): # for each data row in original matrix
            if k % 100000 == 0: 
                gc.collect()
                logging.debug(k)
            start = k * ncols
            end = (k+1) * ncols
            i[start:end] = k # set row indices to current row index
            for l in xrange(ncols):
                j[start+l] = self.value_to_col[l].get(X[k,l], ncols-l-1)
        X_hot = sparse.csr_matrix((data, (i,j)), shape=(nrows, self.ncols), dtype=dtype) 
        return X_hot
        
    def unique_values(self, x, freq_cutoff=0):
        '''
        Return unique values in @x havingn frequency > cutoff
        '''
        # sort values by frequency - note scipy.stats.itemfreq is much slower
        values, inv = np.unique(x, return_inverse=True)
        freq = np.bincount(inv)
        logging.debug("%d unique features" % len(values))
        idx = np.argsort(freq)[::-1]
        values = values[idx]
        freq = freq[idx]
        if freq_cutoff is not None:
            values = values[freq > freq_cutoff]
            logging.debug("%d features retained" % len(values))
        return values