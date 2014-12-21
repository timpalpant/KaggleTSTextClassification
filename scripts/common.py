'''
Utilities for converting raw data (in csv format)
into integer-encoded npz files, scoring predictions,
and saving predictions in submission format
'''

import sys, csv, logging
import numpy as np
from scipy import sparse
from sklearn.cross_validation import KFold

# The type of each feature
B = 0 # boolean
I = 1 # integer
F = 2 # float
S = 3 # string
TYPES = [B, I, F, S]
FEATURE_TYPE = np.asarray([B, B, S, S, F, F, F, F, F, B, 
                           B, B, B, B, I, F, I, I, F, F, 
                           F, I, I, B, B, B, I, F, F, B, 
                           B, B, B, S, S, F, F, F, F, F, 
                           B, B, B, B, B, I, F, I, I, F, 
                           F, F, I, I, B, B, B, I, F, F, 
                           S, B, B, S, S, F, F, F, F, F, 
                           B, B, B, B, B, I, F, I, I, F, 
                           F, F, I, I, B, B, B, I, F, F, 
                           S, B, B, S, S, F, F, F, F, F, 
                           B, B, B, B, B, I, F, I, I, F, 
                           F, F, I, I, B, B, B, I, F, F, 
                           F, F, F, F, F, B, B, B, B, B, 
                           I, F, I, I, F, F, F, I, I, B, 
                           B, B, I, F, F])
BINARY_FEATURES = np.where(FEATURE_TYPE == B)[0]
INT_FEATURES = np.where(FEATURE_TYPE == I)[0]
FLOAT_FEATURES = np.where(FEATURE_TYPE == F)[0]
STRING_FEATURES = np.where(FEATURE_TYPE == S)[0]

# map empty values to -1 / NaN
def truthy(value, col=None):
    if value == 'YES': return 1
    elif value == 'NO': return 0
    else: return -1
def inty(value, col=None):
    if value == '': return -1
    return int(value)
def floaty(value, col=None):
    if value == '': return np.nan
    return float(value)
SMAP = {'': -1} # The hashes are not meaningful: map them to ints
def stringy(value, col=None):
    global SMAP
    v = SMAP.get(value, None)
    if v is None:
        v = len(SMAP)
        SMAP[value] = v
    return v
CONVERTERS = [truthy, inty, floaty, stringy]
FEATURE_CONVERTERS = [CONVERTERS[t] for t in FEATURE_TYPE]

def extract_features(row, cols):
    '''
    Given a @row of training data, extract @cols from the row,
    performing the appropriate data type mapping/conversion
    '''
    return [FEATURE_CONVERTERS[c](row[c], i) 
            for i, c in enumerate(cols)]

def load_labels(csvfile):
    '''
    Load the labels from CSV into dict
    with keys header, ids, labels
    '''
    with open(csvfile) as fd:
        reader = csv.reader(fd)
        header = reader.next()[1:]
        ids = []
        labels = []
        for row in reader:
            row = map(int, row)
            ids.append(row[0])
            labels.append(row[1:])
    ids = np.asarray(ids)
    labels = np.asarray(labels, dtype=np.bool)
    data = {'header': header,
            'ids': ids,
            'labels': labels}
    return data

def load_features(csvfile):
    '''
    Load the features from CSV into dict
    with keys header, ids, bfeatures, ifeatures, ffeatures, sfeatures
    '''
    ids = []
    bfeatures = []
    ifeatures = []
    ffeatures = []
    sfeatures = []
    with open(csvfile) as fd:
        reader = csv.reader(fd)
        header = reader.next()[1:]
        for i, row in enumerate(reader):
            if i % 100000 == 0: logging.debug(i)
            ids.append(int(row[0]))
            row = row[1:]
            bfeatures.append(extract_features(row, BINARY_FEATURES))
            ifeatures.append(extract_features(row, INT_FEATURES))
            ffeatures.append(extract_features(row, FLOAT_FEATURES))
            sfeatures.append(extract_features(row, STRING_FEATURES))
    ids = np.asarray(ids)
    bfeatures = np.asarray(bfeatures, dtype=np.int8)
    ifeatures = np.asarray(ifeatures, dtype=np.int16)
    ffeatures = np.asarray(ffeatures)
    sfeatures = np.asarray(sfeatures)
    data = {'header': header,
            'ids': ids,
            'bfeatures': bfeatures,
            'ifeatures': ifeatures,
            'ffeatures': ffeatures,
            'sfeatures': sfeatures}
    return data
    
def guess_loader(f):
    '''Guess the appropriate function to load file @f'''
    if f.endswith('.npz'): return load_npz
    with open(f) as fd:
        header = fd.readline()
    if 'x' in header: return load_features
    if 'y' in header: return load_labels
    
def load_npz(filename):
    return np.load(filename)
    
def save_npz(filename, *args, **data):
    np.savez(filename, *args, **data)
    
def load_encoded_features(filename):
    loader = load_npz(filename)
    try: # reconstruct sparse arrays
        X = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                              shape=loader['shape'])
    except: # dense arrays
        X = loader['X']
    return X
    
def save_encoded_features(filename, X):
    try: # sparse arrays
        np.savez(filename, data=X.data, indices=X.indices, 
                 indptr=X.indptr, shape=X.shape)
    except: # dense arrays
        np.savez(filename, X=X)
    
def save_predictions(ids, labels, pred, fd):
    '''Write predictions @pred in submission format to @fd'''
    writer = csv.writer(fd)
    header = ('id_label', 'pred')
    writer.writerow(header)
    for i, id in enumerate(ids):
        if i % 10000 == 0: print i
        for j, label in enumerate(labels):
            row = ('%d_%s' % (id, label), pred[i,j])
            writer.writerow(row)
    
LOW_CLAMP = 1e-15
HIGH_CLAMP = 1 - 1e-15
    
def clamp(x, low=LOW_CLAMP, high=HIGH_CLAMP, copy=False):
    '''clamp @x to within low-high (in place if copy=False)'''
    x = np.array(x, copy=copy)
    x[x<low] = low
    x[x>high] = high
    return x
    
def score_predictions(ref, pred, axis=None):
    '''score predicted labels @pred vs. gold standard @ref'''
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    pred = clamp(pred)
    ref = clamp(ref)
    score = -np.mean(ref*np.log(pred) + (1-ref)*np.log(1-pred), axis=axis)
    return score
    
def cross_validate(clf, X, Y, n_folds=3):
    N = X.shape[0]
    kf = KFold(N, n_folds=n_folds, random_state=42)
    scores = []
    for train, test in kf:
        X_train, X_test = X[train], X[test]
        Y_train, Y_test = Y[train], Y[test]
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        score = score_predictions(Y_test, Y_pred, axis=0)
        logging.info("Cross-validation score: %f" % score.mean())
        scores.append(score)
    return np.asarray(scores)
    
def load_predictions(fd):
    reader = csv.reader(fd)
    header = reader.next()
    last_id = None
    ids = np.arange(1700001, 2245083)
    n = len(ids)
    Y = np.zeros((n, 33))
    for row in reader:
        id, label_id = row[0].split('_')
        id = int(id) - 1700001
        label_id = int(label_id[1:]) - 1
        Y[id,label_id] = float(row[1])
    pred = {'header': ['y%d' % i for i in xrange(1,34)],
            'ids': ids,
            'labels': np.asarray(Y)}
    return pred
    
def optional(a):
    '''are there empty values in @a? (per column)'''
    return np.logical_or(np.any(np.isnan(a), axis=0),
                         np.any(a == -1, axis=0))
