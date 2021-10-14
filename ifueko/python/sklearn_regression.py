# Adapted from https://coax.readthedocs.io/en/latest/examples/linear_regression/jax.html
import jax
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def loss(y, pred):
  error = pred - y
  return np.mean(np.power(error, 2))

# create our dataset
X, y = make_regression(n_features=3, random_state=0)
X, X_test, y, y_test = train_test_split(X, y)

reg = LinearRegression()
reg.fit(X, y)
print("Train", loss(y, reg.predict(X)))
print("Test", loss(y_test, reg.predict(X_test)))
