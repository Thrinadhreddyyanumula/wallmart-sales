# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 08:04:11 2023

@author: 91955
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot
as
plt
data=pd.read_csv(r"C:\Users\91955\AppData\Local\Temp\Rar$DIa20360.1889/Train.csv&quot
;)
data.columns
y=data.iloc[:,-1]
X=data.iloc[:,5]
X = X.values.reshape(-1, 1)
from
sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =
train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.linear_model import
LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred =
model.predict(X_test)
from sklearn.metrics import mean_squared_error,
mean_absolute_error
mse = mean_squared_error(y_test, y_pred)
mae =
mean_absolute_error(y_test, y_pred)
print("Mean Squared Error:",
mse)
print("Mean Absolute Error:", mae)
from sklearn.metrics import
mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# Calculate
R-squared
r2 = r2_score(y_test, y_pred)
# Print the evaluation
metrics
print("RMSE:", rmse)
print("R-squared:", r2)
from sklearn.tree
import DecisionTreeRegressor
model = DecisionTreeRegressor (max_depth=5,
min_samples_split=2,min_samples_leaf=1,max_features=None)
model.fit(X_train, y_train)
y_pred
= model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 =
r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("R-squared:",
r2)
Mean Squared Error: 2007708.54288217
Mean Absolute Error:
1061.7852566570666
RMSE: 1416.9363228042996
R-squared: 0.31403998885857476
RMSE:
1432.5820433642593
R-squared: 0.2988077001867605
Powered by TCPDF (www.tcpdf.org)