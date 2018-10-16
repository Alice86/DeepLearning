#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 232A - Project 1
 Author: Jiayu WU
 Date : 0114018
 Description: Codes for Project 2
"""
import os
os.getcwd() 
os.chdir("/Users/alice/Documents/Projects/232/project2") 

"""
cd /Users/alice/Documents/Projects/232/project2
source activate py36
conda install -c anaconda future 
conda install --yes --file requirements.txt
"""

"""
! conda install -c anaconda wget 
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install wget
cd ./stats232a/datasets
./get_datasets.sh
"""

from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from stats232a.classifiers.fc_net import *
from stats232a.data_utils import get_mnist_data
from stats232a.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from stats232a.solver import Solver
from stats232a.layers import *

% matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2

def rel_error(x, y):
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Load the (preprocessed) MNIST data.
# The second dimension of images indicated the number of channel. For black and white images in MNIST, channel=1.
data = get_mnist_data()
for k, v in list(data.items()):
  print(('%s: ' % k, v.shape))


"""
Forward pass for a fully-connected layer.
- x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
- w: A numpy array of weights, of shape (D, M)
- b: A numpy array of biases, of shape (M,)
Returns a tuple of:
- out: output, of shape (N, M)
- cache: (x, w, b)
"""
def fc_forward(x, w, b):
    pass # The pass statement is a null operation
    
    N, D = x.shape[0], w.shape[0]
    xs = x.reshape(N, D)
    out = np.dot(xs, w)+b 

    cache = (x, w, b)
    return out, cache  
  
# Test the fc_forward function
num_inputs = 2
input_shape = (4, 5, 6)
output_dim = 3
input_size = num_inputs * np.prod(input_shape)
weight_size = output_dim * np.prod(input_shape)
x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
b = np.linspace(-0.3, 0.1, num=output_dim)

out, _ = fc_forward(x, w, b)
correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                        [ 3.25553199,  3.5141327,   3.77273342]])
# Compare your output with ours. The error should be around 1e-9.
print('Testing fc_forward function:')
print('difference: ', rel_error(out, correct_out))



"""
Backward pass for a fully-connected layer.
- dout: Upstream derivative, of shape (N, M)
- cache: Tuple of:
- x: Input data, of shape (N, d_1, ... d_k)
- w: Weights, of shape (D, M)
Returns a tuple of:
- dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
- dw: Gradient with respect to w, of shape (D, M)
- db: Gradient with respect to b, of shape (M,)
"""
def fc_backward(dout, cache):

    x, w, b = cache
    xs = x.reshape(x.shape[0], w.shape[0])
    
    db = np.sum(dout, axis = 0) # by col, M
    dw = np.dot(xs.T, dout)  # D features * M channels
    dx = np.dot(dout, w.T).reshape(x.shape) # N obs. * D features
  
    return dx, dw, db


# Test the fc_backward function
np.random.seed(231)
x = np.random.randn(10, 2, 3)
w = np.random.randn(6, 5)
b = np.random.randn(5)
dout = np.random.randn(10, 5)
dx_num = eval_numerical_gradient_array(lambda x: fc_forward(x, w, b)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: fc_forward(x, w, b)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: fc_forward(x, w, b)[0], b, dout)
_, cache = fc_forward(x, w, b)
dx, dw, db = fc_backward(dout, cache)
# The error should be around 1e-10
print('Testing fc_backward function:')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))


"""
Computes the forward pass for a layer of rectified linear units (ReLUs).
- x: Inputs, of any shape
Returns a tuple of:
- out: Output, of the same shape as x
- cache: x
"""
def relu_forward(x):

    # out = np.fmax(x, np.zeros(x.shape))
    out = np.copy(x)
    out[out<0] = 0
    cache = x
    
    return out, cache

# Test the relu_forward function
x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
out, _ = relu_forward(x)
correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
                        [ 0.,          0.,          0.04545455,  0.13636364,],
                        [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])
# Compare your output with ours. The error should be around 5e-8
print('Testing relu_forward function:')
print('difference: ', rel_error(out, correct_out))


"""
Computes the backward pass for a layer of rectified linear units (ReLUs).
- dout: Upstream derivatives, of any shape
- cache: Input x, of same shape as dout
Returns:
- dx: Gradient with respect to x
"""
def relu_backward(dout, cache):

    dx, x = None, cache
    dx = np.copy(dout)
    dx[x<0] = 0
    return dx

# Test the relu_backward function
np.random.seed(231)
x = np.random.randn(10, 10)
dout = np.random.randn(*x.shape)
dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)
_, cache = relu_forward(x)
dx = relu_backward(dout, cache)
# The error should be around 3e-12
print('Testing relu_backward function:')
print('dx error: ', rel_error(dx_num, dx))


"""
"Sandwich" layers
- x: Input to the affine layer
- w, b: Weights for the affine layer
Returns a tuple of:
- out: Output from the ReLU
- cache: Object to give to the backward pass
"""
from stats232a.layers import *

def fc_relu_forward(x, w, b):
    outs, fc = fc_forward(x, w, b)
    out, rl = relu_forward(outs)
    cache = (fc, rl)
    return out, cache # computed values for hidden layer, input from previous(original & relued) for bp 

def fc_relu_backward(dout, cache): 
    fc, rl = cache      
    dout = relu_backward(dout, rl)
    dx, dw, db = fc_backward(dout, fc)   
    return dx, dw, db # gradients computed

# numerically gradient check the backward pass:
# from stats232a.layer_utils import fc_relu_forward, fc_relu_backward
np.random.seed(231)
x = np.random.randn(2, 3, 4)
w = np.random.randn(12, 10)
b = np.random.randn(10)
dout = np.random.randn(2, 10)
out, cache = fc_relu_forward(x, w, b)
dx, dw, db = fc_relu_backward(dout, cache)
dx_num = eval_numerical_gradient_array(lambda x: fc_relu_forward(x, w, b)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: fc_relu_forward(x, w, b)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: fc_relu_forward(x, w, b)[0], b, dout)
print('Testing affine_relu_forward:')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))



"""
Computes the loss and gradient for softmax classification.
- x: Input data, of shape (N, C) where x[i, j] is the score for the jth
  class for the ith input.
- y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
  0 <= y[i] < C
Returns a tuple of:
- loss: Scalar giving the loss
- dx: Gradient of the loss with respect to x
"""  
from builtins import range  
def softmax_loss(x, y):
    N = x.shape[0]
    a = np.vstack(np.max(x, axis=1))
    prs = np.exp(x-a)
    pr = prs/np.vstack(np.sum(prs, axis = 1))    # *exp(a)
    loss = np.sum(-np.log(pr[range(N), y]))/N # why /N, y is labels/categorical response
    dx = np.copy(pr)
    dx[range(N),y] -= 1
    dx /= N
    
    return loss, dx

# Test softmax_loss function.
np.random.seed(231)
num_classes, num_inputs = 10, 50
x = 0.001 * np.random.randn(num_inputs, num_classes)
y = np.random.randint(num_classes, size=num_inputs)
dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
loss, dx = softmax_loss(x, y)
# Loss should be 2.3 and dx error should be 1e-8
print('\nTesting softmax_loss:')
print('loss: ', loss)
print('dx error: ', rel_error(dx_num, dx))



"""
First-order update rules for training neural networks
Each update rule accepts current weights & the gradient of the loss with respect to those weights
to produces the next set of weights

Interface:
def update(w, dw, config=None):
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.
Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""

"""
A Solver encapsulates all the logic necessary for training classification
models. 

Performs stochastic gradient descent using different update rules defined in optim.py.

Accepts both training and validataion data and labels so it can
periodically check classification accuracy on both training and validation
data to watch out for overfitting.

To train a model, you will first construct a Solver instance, passing the
model, dataset, and various optoins (learning rate, batch size, etc) to the
constructor. You will then call the train() method to run the optimization
procedure and train the model.
After the train() method returns, model.params will contain the parameters
that performed best on the validation set over the course of training.
In addition, the instance variable solver.loss_history will contain a list
of all losses encountered during training and the instance variables
solver.train_acc_history and solver.val_acc_history will be lists of the
accuracies of the model on the training and validation set at each epoch.

Example:
data = {
  'X_train': # training data
  'y_train': # training labels
  'X_val': # validation data
  'y_val': # validation labels
}
model = MyAwesomeModel(hidden_size=100, reg=10)
solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 1e-3,
                },
                lr_decay=0.95,
                num_epochs=10, batch_size=100,
                print_every=100)
solver.train()
“”“