# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:14:28 2019

@author: sun_y
"""

from sklearn.datasets import load_iris
from sklearn import tree
import sys
import os       
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'



## load data
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)


## store the model in dot file
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
    
## visualize the decision tree

from IPython.display import Image  
import pydotplus 
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 
