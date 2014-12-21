#!/usr/bin/env python

'''
Make predictions for the test data

6. Use logistic regression
'''

import argparse, multiprocessing
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def prepare_features(data, enc=None, scaler=None):
    '''
    One-hot encode all boolean/string (categorical) features,
    and shift/scale integer/float features
    '''
    # X needs to contain only non-negative integers
    bfs = data['bfeatures'] + 1
    sfs = data['sfeatures'] + 1
    
    # Shift/scale integer and float features to have mean=0, std=1
    ifs = data['ifeatures']
    ffs = data['ffeatures']
    x2 = np.hstack((ifs,ffs))
    if scaler is None:
        scaler = StandardScaler()
        x2 = scaler.fit_transform(x2)
        print "Training features have mean: %s" % scaler.mean_
        print "and standard deviation: %s" % scaler.std_
    else:
        x2 = scaler.transform(x2, copy=False)
        
    # one-hot encode categorical features
    X = np.hstack((bfs,sfs,x2))
    categorical = np.arange(bfs.shape[1]+sfs.shape[1])
    if enc is None:
        enc = OneHotEncoder(n_values='auto', categorical_features=categorical)
        X = enc.fit_transform(X)
        print "One-hot encoded features have dimension %d" % X.shape[1]
    else:
        X = enc.transform(X)
    return X, enc, scaler

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=load_npz,
        help='Training features (npz)')
    parser.add_argument('labels', type=load_npz,
        help='Training labels (npz)')
    parser.add_argument('test', type=load_npz,
        help='Test features (npz)')
    parser.add_argument('output',
        help='Output label predictions (npz)')
    return parser
    
def cross_validate_classifier(clf, features, labels, cv=5):
    kf = cross_validation.KFold(len(features), n_folds=cv)
    scores = []
    for train, test in kf:
        clf.fit(features[train], labels[train])
        Y = np.vstack([v[:,0] for v in clf.predict_proba(features[test])])
        pred = 1 - Y.T
        score = score_predictions(labels[test], pred)
    return scores

if __name__ == "__main__":
    args = opts().parse_args()
    
    print "Loading and preparing data"
    X, enc, scaler = prepare_features(args.train)
    Y = args.labels['labels'][:1111]
    label_header = args.labels['header']
    
    print "Training classifier"
    clfs = [LogisticRegression(C=1.0, tol=0.001, random_state=42) for y in Y.T]
    for i, (clf, y) in enumerate(zip(clfs, Y.T)): 
        print "Fitting label %s" % label_header[i]
        try: clf.fit(X, y)
        except: clfs[i] = y[0]
    del X, Y
    
    print "Predicting"
    X = prepare_features(args.test, enc, scaler)
    p = []
    for i, clf in enumerate(clfs):
        print "Predicting label %s" % label_header[i]
        print clf.predict_proba(X)
        try: p.append(clf.predict_proba(X)[:,0])
        except: p.append(np.array([clf] * len(X)))
    p = np.vstack(p).T
    Y = 1 - p
    
    print "Saving predictions"
    save_npz(args.output, ids=args.test['ids'], 
        header=args.labels['header'], labels=Y)
    
