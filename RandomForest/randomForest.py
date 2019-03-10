# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:56:12 2019

@author: sun_y
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics

import matplotlib.pylab as plt

## load data
train = pd.read_csv('train_modified.csv')
target='Disbursed' 
IDcol = 'ID'
train['Disbursed'].value_counts() 
x_columns = [x for x in train.columns if x not in [target, IDcol]]
X = train[x_columns]
y = train['Disbursed']

### train model
rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(X,y)
print (rf0.oob_score_)
y_predprob = rf0.predict_proba(X)[:,1]
print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))



### find the best parameters

## A : estimAtors
param_test1 = {'n_estimators':list(range(20,81,10))}
gsearch1 = GridSearchCV(estimator = 
                        RandomForestClassifier(oob_score=True, 
                                                   min_samples_split=300,
                                                   min_samples_leaf=20,
                                                   max_depth=8,
                                                   max_features='sqrt', 
                                                   random_state=10), 
                                                   param_grid = param_test1,
                                                   scoring='roc_auc',
                                                   iid=False,
                                                   cv=5)
gsearch1.fit(X,y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


## B: max depth
param_test2 = {'max_depth':list(range(3,14,2)),
               'min_samples_split':list(range(100,801,200))}
gsearch2 = GridSearchCV(estimator = 
                        RandomForestClassifier(oob_score=True,
                                                   n_estimators=60, 
                                                   min_samples_leaf=20, 
                                                   max_features='sqrt', 
                                                    
                                                   random_state=10), 
                                                   param_grid = param_test2, 
                                                   scoring='roc_auc',
                                                   iid=False, 
                                                   cv=5)
gsearch2.fit(X,y)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

### C: leaf and sample

param_test3 = {'min_samples_split':list(range(800,1900,200)),
               'min_samples_leaf':list(range(60,101,10))}
gsearch3 = GridSearchCV(estimator = 
                        RandomForestClassifier(oob_score=True, 
                                                   n_estimators=60,
                                                   max_depth=7,
                                                   max_features='sqrt', 
                                                    
                                                   random_state=10), 
                                                   param_grid = param_test3, 
                                                   scoring='roc_auc',
                                                   iid=False, 
                                                   cv=5)
gsearch3.fit(X,y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


### D: max features
param_test4 = {'max_features':list(range(7,20,2))}
gsearch4 = GridSearchCV(estimator = 
                        RandomForestClassifier(oob_score=True,
                                                   n_estimators=60,
                                                   max_depth=7, 
                                                   min_samples_leaf =60, 
                                                   min_samples_split =1200, 
                                                    
                                                   random_state=10), 
                                                   param_grid = param_test4, 
                                                   scoring='roc_auc',
                                                   iid=False, 
                                                   cv=5)
gsearch4.fit(X,y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


rf2 = RandomForestClassifier(n_estimators= 60, 
                             max_depth=13, 
                             min_samples_split=120,
                             min_samples_leaf=20,
                             max_features=7 ,
                             oob_score=True, 
                             random_state=10)
rf2.fit(X,y)
print (rf2.oob_score_)