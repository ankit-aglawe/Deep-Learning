from sklearn.datasets import load_boston
from sklearn.neural_network import MLPRegressor
import numpy as np

boston = load_boston()

X = boston.data
y = boston.target

mlp = MLPRegressor()

mlp.fit(X,y)

y_pred = mlp.predict(X)

#print(y_pred)

#from sklearn.metrics import mean_absolute_error, mean_sqaured_error

from sklearn import metrics

print(metrics.mean_absolute_error(y,y_pred))
print(metrics.mean_squared_error(y,y_pred))
print(np.sqrt(metrics.mean_squared_error(y,y_pred)))
