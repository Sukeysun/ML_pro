# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:25:39 2019

@author: sun_y
"""

import matplotlib.pyplot as plt
# matplotlib inline
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model


# read data and split X and y
data = pd.read_csv('.\data\Folds5x2_pp.csv')

X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]

### split training set and testing set 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

#### using lr model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import RidgeCV,Ridge

## find a best alpha
ridgecv = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100])
ridgecv.fit(X_train, y_train)
ridgecv.alpha_ 

ridge = Ridge(alpha = ridgecv.alpha_)
ridge.fit(X_train, y_train)

## predict y in two ways
y_pred = ridge.predict(X_test)


## evaluate the model
from sklearn import metrics
# calculate MSE
print ("MSE:",metrics.mean_squared_error(y_test, y_pred))
# RMSE
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()