# Jax Adapted from https://coax.readthedocs.io/en/latest/examples/linear_regression/jax.html
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def make_dataset(n_features=3, random_state=0, n_samples=100):
  X, y = make_regression(n_features=n_features, random_state=random_state, n_samples=n_samples)
  X, X_test, y, y_test = train_test_split(X, y)
  return X, X_test, y, y_test

def loss_np(y, pred):
  error = pred - y
  return np.mean(np.power(error, 2))

def regression_sklearn(X, y):
  reg = LinearRegression()
  reg.fit(X, y)
  return reg

def evaluate_sklearn(reg, X, y):
  print("loss (sklearn): ", loss_np(y, reg.predict(X)))

@jax.jit
def loss_jax(params, X, y):
    err = forward(params, X) - y
    return jnp.mean(jnp.square(err))  # mse

@jax.jit
def forward(params, X):
    return jnp.dot(X, params['w']) + params['b']

@jax.jit
def update(params, grads):
    return jax.tree_multimap(lambda p, g: p - 0.05 * g, params, grads)

grad_fn = jax.grad(loss_jax)

@jax.jit
def regression_jax(X, y):
  params = {
    'w': jnp.zeros(X.shape[1:]),
    'b': 0.
  }
  for _ in range(200):
    grads = grad_fn(params, X, y)
    params = update(params, grads)
  return params

def evaluate_jax(params, X, y):
  print("loss (jax):     ", loss_jax(params, X, y))
