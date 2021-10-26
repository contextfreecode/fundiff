# Adapted from https://github.com/google/jax/blob/main/examples/mnist_classifier.py
import time
import itertools
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datasets
import jax.numpy as jnp
from jax import jit, grad, random
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax
from jax.experimental import optimizers

def loss_torch(model, batch):
  inputs, targets = batch
  preds = model(inputs)
  return -torch.mean(torch.sum(preds * targets, axis=1))

def accuracy_pytorch(model, batch):
  inputs, targets = batch
  target_class = torch.argmax(targets, axis=1)
  predicted_class = torch.argmax(model(inputs), axis=1)
  return torch.mean((predicted_class == target_class).float())

class MNISTClassifier(nn.Module):
  def __init__(self, weights):
    super(MNISTClassifier, self).__init__()
    self.d1 = nn.Linear(28*28, 1024)
    self.d2 = nn.Linear(1024, 1024)
    self.d3 = nn.Linear(1024, 10)
    self.d1.weight.data = weights[0]
    self.d1.bias.data = weights[1]
    self.d2.weight.data = weights[2]
    self.d2.bias.data = weights[3]
    self.d3.weight.data = weights[4]
    self.d3.bias.data = weights[5]

  def forward(self, x):
    x = self.d1(x)
    x = F.relu(x)
    x = self.d2(x)
    x = F.relu(x)
    x = self.d3(x)
    x = F.log_softmax(x, dim=0)
    return x

def make_dataset(pytorch=False):
  train_images, train_labels, test_images, test_labels = datasets.mnist(use_torch=pytorch)
  return train_images, train_labels, test_images, test_labels

def initialize_pytorch():
  rng = random.PRNGKey(0)
  init_random_params, predict = stax.serial(
      Dense(1024), Relu,
      Dense(1024), Relu,
      Dense(10), LogSoftmax)
  _, init_params = init_random_params(rng, (-1, 28 * 28))
  torch_weights = []
  for item in init_params:
    for m in item:
      m = np.asarray(m).T.copy()
      torch_weights.append(torch.FloatTensor(m))
  classifier = MNISTClassifier(torch_weights).cuda()
  return classifier

step_size = 0.001
num_epochs = 10
batch_size = 128
momentum_mass = 0.9

def train_pytorch(classifier, train_images, train_labels):
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)
  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]
  batches = data_stream()
  optimizer = optim.SGD(classifier.parameters(), lr=step_size, momentum=momentum_mass)
  def update(model, batch):
    optimizer.zero_grad()
    curr_loss = loss_torch(model, batch)
    curr_loss.backward()
    optimizer.step() 
  for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
      update(classifier, next(batches))
    epoch_time = time.time() - start_time
  return classifier

def evaluate_pytorch(classifier, images, labels):
    acc = accuracy_pytorch(classifier, (images, labels))
    print("Accuracy (Pytorch) {}".format(acc))


def loss_jax(params, predict, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))

def accuracy_jax(params, predict, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(predict(params, inputs), axis=1)
  return jnp.mean(predicted_class == target_class)

def initialize_jax():
  rng = random.PRNGKey(0)
  init_random_params, predict = stax.serial(
      Dense(1024), Relu,
      Dense(1024), Relu,
      Dense(10), LogSoftmax)
  _, init_params = init_random_params(rng, (-1, 28 * 28))
  return init_params, predict


def train_jax(model, train_images, train_labels):
  model_params, predict = model
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]
  batches = data_stream()

  opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

  @jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss_jax)(params, predict, batch), opt_state)

  opt_state = opt_init(model_params)
  itercount = itertools.count()
  for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
      opt_state = update(next(itercount), opt_state, next(batches))
    epoch_time = time.time() - start_time

    params = get_params(opt_state)
  return params, predict

def evaluate_jax(model, images, labels):
    params, predict = model
    acc = accuracy_jax(params, predict, (images, labels))
    print("Accuracy (Jax)     {}".format(acc))
