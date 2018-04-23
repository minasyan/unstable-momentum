import numpy as np
import torch
from torch.autograd import Variable

'''
SGD Implementation for a decomposable objective loss function. This is the vanilla SGD implementation.

Input: w - current parameter vector
       f - loss function
       x - training point
       y - output vector
       alpha - learning rate

Output: result found by SGD on the dataset
'''
def sgd_step(w, f, x, y, alpha):
    loss = f(w, x, y)
    loss.backward()
    new_w = w - alpha * w.grad
    return new_w

'''
Momentum SGD implementation for a decomposable objective loss function.

Input: w - current parameter vector
       f - loss function
       x - training point
       y - output vector
       alpha - learning rate
       beta - momentum averaging rate

Output: result found by momentum SGD on the dataset
'''
def msgd_step(w, w_prev, f, X, y, alpha, beta):
    loss = f(w, x, y)
    loss.backward()
    new_w = w - alpha * w.grad + beta * (w - w_prev)
    return new_w
