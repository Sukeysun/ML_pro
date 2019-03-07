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
linreg = LinearRegression()
linreg.fit(X_train, y_train)

print (linreg.intercept_)
print (linreg.coef_)

## predict y in two ways
y_pred = linreg.predict(X_test)

predicted = cross_val_predict(linreg, X, y, cv=10)

## evaluate the model
from sklearn import metrics
# calculate MSE
print ("MSE:",metrics.mean_squared_error(y_test, y_pred))
# RMSE
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# calculate MSE
print ("MSE:",metrics.mean_squared_error(y, predicted))
# RMSE
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted)))

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()