# Adapted from https://coax.readthedocs.io/en/latest/examples/linear_regression/jax.html
import jax
import jax.numpy as jnp
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


# create our dataset
X, y = make_regression(n_features=3, random_state=0)
X, X_test, y, y_test = train_test_split(X, y)


# model weights
params = {
    'w': jnp.zeros(X.shape[1:]),
    'b': 0.
}


def forward(params, X):
    return jnp.dot(X, params['w']) + params['b']


def loss_fn(params, X, y):
    err = forward(params, X) - y
    return jnp.mean(jnp.square(err))  # mse


grad_fn = jax.grad(loss_fn)


def update(params, grads):
    return jax.tree_multimap(lambda p, g: p - 0.05 * g, params, grads)


# the main training loop
for _ in range(200):
    grads = grad_fn(params, X, y)
    params = update(params, grads)
loss_train = loss_fn(params, X, y)
loss_test = loss_fn(params, X_test, y_test)
print('Train', loss_train)
print('Test', loss_test)
