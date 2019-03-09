# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:27:00 2019

@author: sun_y
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles


## generate data
X1, y1 = make_gaussian_quantiles(cov=2.0,n_samples=500, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,n_samples=400, 
                                 n_features=2, n_classes=2, random_state=1)
## concatenate data
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))