'''
Classifiers for the Tradeshift Text Classification challenge

@author Timothy Palpant <tim@palpant.us>
@date October 17, 2014
'''

import logging
from common import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, log_loss
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

class TSClassifier(object):
    def classifier(self):
        raise RuntimeError("No classifier implemented")
    
    def fit(self, X, y):
        self.clf = self.classifier()
        self.y_mean = y.mean()
        try: 
            self.clf.fit(X, y)
            logging.info("Fit params: %s" % self.clf.get_params())
        except ValueError, e:
            logging.warning(e)
            self.clf = None
    
    def predict(self, X):
        try:
            return self.clf.predict_proba(X)[:,1]
        except:
            return self.y_mean * np.ones(X.shape[0])

class TSCategoricalNBClassifier(TSClassifier):
    '''
    A Bernoulli naive bayes classifier
    ''' 
    def classifier(self):
        return BernoulliNB(fit_prior=True)
    
class TSLogisticRegressionClassifier(TSClassifier):
    '''
    A logistic regression classifier
    '''
    fit_hyperparameters = True
    param_grid = [{'C': [0.1, 1.0, 10.]}]
    cv_fit = 3
    
    def classifier(self):
        clf = LogisticRegression(C=1.0, tol=0.001, random_state=42)
        if self.fit_hyperparameters:
            scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
            clf = GridSearchCV(clf, self.param_grid, scoring=scorer, 
                               n_jobs=1, cv=self.cv_fit, verbose=2)
        return clf

class TSSGDClassifier(TSClassifier):
    '''
    A logistic regression classifier fit with SGD
    '''
    fit_hyperparameters = False
    param_grid = [{'alpha': [0.0001, 0.001, 0.005]}]
    cv_fit = 3
    n_iter = 10
    
    def classifier(self):
        clf = SGDClassifier(loss='log', n_iter=self.n_iter, alpha=0.1, 
            n_jobs=1, verbose=2, random_state=42)
        if self.fit_hyperparameters:
            scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
            clf = GridSearchCV(clf, self.param_grid, scoring=scorer, 
                               n_jobs=1, cv=self.cv_fit, verbose=2)
        return clf
    
class TSRandomForestClassifier(TSClassifier):
    '''
    A RandomForest model
    ''' 
    criterion = 'entropy'
    n_estimators = 48
    
    def classifier(self):
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            min_samples_split=1, max_depth=None,
            criterion=self.criterion,
            n_jobs=-1, random_state=42, verbose=2)
        return clf

class TSSVCRandomForestClassifier(TSClassifier):
    '''
    A RandomForest model with features first selected by LinearSVC
    ''' 
    criterion = 'entropy'
    n_estimators = 48
    
    def classifier(self):
        clf = Pipeline([
          ('feature_selection', LinearSVC(penalty="l1", dual=False)),
          ('classification', RandomForestClassifier(
                n_estimators=self.n_estimators, 
                min_samples_split=1, max_depth=None,
                criterion=self.criterion,
                n_jobs=-1, random_state=42, verbose=2))
        ])
        return clf
    
class TSEnsembleClassifier(TSClassifier):
    '''
    A classifier that prepares features and calls sub-classifiers.
    The predictions of each of the sub-classifiers are averaged.
    '''
    # a list of 2-tuples: [(classifier, labels)]
    # each method will make a prediction for each label in labels, 
    # and the predictions will be blended
    methods = [(TSLogisticRegressionClassifier, None)]
    
    def fit(self, X, Y):
        # Get predictions from each individual classifier
        logging.info("Fitting individual classifiers")
        self.classifiers = []
        self.nlabels = labels.shape[1]
        for classifier, columns in self.methods:
            clfs = []
            if columns is None: # predict all labels
                columns = np.arange(self.nlabels)
            for i in columns:
                y = Y[:,i]
                clf = classifier()
                clf.fit(X, y)
                clfs.append((i,clf))
            self.classifiers.append(clfs)
            del X
        
    def predict(self, X):
        # Get predictions from each individual classifier
        logging.info("Predicting with individual classifiers")
        self.Xp = [list() for i in xrange(self.nlabels)]
        for enc, clfs in self.classifiers:
            for i, clf in clfs:
                y = clf.predict(X)
                self.Xp[i].append(y)
            del X
        self.Xp = [np.vstack(Xi).T for Xi in self.Xp]
            
        # Combine individual classifier predictions according to weights
        logging.info("Weighting and combining predictions")
        Y = [X_i.mean(axis=1) for X_i in self.Xp]
        Y = np.hstack(Y)
        return Y