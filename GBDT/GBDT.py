# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:39:35 2019

@author: sun_y
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt


## load data
train = pd.read_csv('train_modified.csv')
target='Disbursed' 
IDcol = 'ID'
train['Disbursed'].value_counts()
x_columns = [x for x in train.columns if x not in [target, IDcol]]
X = train[x_columns]
y = train['Disbursed']


## use all default value to train model and check the performance

gbm0 = GradientBoostingClassifier(random_state=10)
gbm0.fit(X,y)
y_pred = gbm0.predict(X)
y_predprob = gbm0.predict_proba(X)[:,1]

###############################
#####Accuracy : 0.9852
###AUC Score (Train): 0.900531
###############################
print ("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

### find the best parameters

## A : estimAtors
param_test1 = {'n_estimators':list(range(20,81,10))}
gsearch1 = GridSearchCV(estimator = 
                        GradientBoostingClassifier(learning_rate=0.1, 
                                                   min_samples_split=300,
                                                   min_samples_leaf=20,
                                                   max_depth=8,
                                                   max_features='sqrt', 
                                                   subsample=0.8,random_state=10), 
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
                        GradientBoostingClassifier(learning_rate=0.1,
                                                   n_estimators=60, 
                                                   min_samples_leaf=20, 
                                                   max_features='sqrt', 
                                                   subsample=0.8, 
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
                        GradientBoostingClassifier(learning_rate=0.1, 
                                                   n_estimators=60,
                                                   max_depth=7,
                                                   max_features='sqrt', 
                                                   subsample=0.8, 
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
                        GradientBoostingClassifier(learning_rate=0.1,
                                                   n_estimators=60,
                                                   max_depth=7, 
                                                   min_samples_leaf =60, 
                                                   min_samples_split =1200, 
                                                   subsample=0.8, 
                                                   random_state=10), 
                                                   param_grid = param_test4, 
                                                   scoring='roc_auc',
                                                   iid=False, 
                                                   cv=5)
gsearch4.fit(X,y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


#### decrease learning rate and increase number of estimators
### train a new model

gbm_new = GradientBoostingClassifier(
        learning_rate=0.01, 
        n_estimators=600,
        max_depth=7, 
        min_samples_leaf =60, 
        min_samples_split =1200, 
        max_features=9, 
        subsample=0.7, 
        random_state=10)

gbm_new.fit(X,y)
y_pred = gbm_new.predict(X)
y_predprob = gbm_new.predict_proba(X)[:,1]
print ("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))