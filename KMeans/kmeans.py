# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:01:52 2019

@author: sun_y
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

## generate data
X, y = make_blobs(n_samples=1000, n_features=2, 
                  centers=[[-1,-1], [0,0], [1,1], [2,2]], 
                  cluster_std=[0.4, 0.2, 0.2, 0.2], 
                  random_state =9)
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()


from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn import metrics
for index, k in enumerate((2,3,4,5)):
    plt.subplot(2,2,index+1)
    y_pred = MiniBatchKMeans(n_clusters=k, batch_size = 200, random_state=9).fit_predict(X)
    score= metrics.calinski_harabaz_score(X, y_pred)  
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.text(.99, .01, ('k=%d, score: %.2f' % (k,score)),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
plt.show()