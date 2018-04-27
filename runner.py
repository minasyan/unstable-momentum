import torch
from torch.autograd import Variable
import numpy as np
import descent
from tqdm import tqdm
from objectives import square_loss
from data_generation import lin_gen, pol_gen, fried1_gen
from matplotlib import pyplot as plt


'''
Runs SGD to comput the needed stability metrics. It computes the Euclidean parametric
distance and the difference between losses at each epoch.

Input: X - feature matrix (numpy array)
       y - output vetor (numpy array)
       f - loss function
       epochs - number of epochs to run on
       alpha - learning rate
'''
def sgd_stability(X, y, f, epochs, alpha):
    ## Create the differring datasets for each run.
    X1, y1, X2, y2  = data_split(X,y)
    n, d = X1.shape
    X1, y1, X2, y2 = Variable(torch.Tensor(X1), requires_grad=True), Variable(torch.Tensor(y1), requires_grad=True),Variable(torch.Tensor(X2), requires_grad=True), Variable(torch.Tensor(y2), requires_grad=True)
    ## Initialize parameters.
    w1 = Variable(torch.Tensor(initialize(d)), requires_grad=True)
    w2 = w1
    loss_distance = [0 for _ in range(epochs)]
    param_distance = [0 for _ in range(epochs)] ## We use L2-norm here
    loss_distance[0] = torch.abs(compute_loss(w1, X1, y1, f) - compute_loss(w2, X2, y2, f)).data.numpy()
    param_distance[0] = torch.norm(w1 - w2).data.numpy()
    for epoch in tqdm(range(1, epochs)):
        for _ in range(n):
            index = np.random.randint(0,n)
            w1 = descent.sgd_step(w1, f, X1[index], y1[index], alpha)
            w2 = descent.sgd_step(w2, f, X2[index], y2[index], alpha)
        loss_distance[epoch] = torch.abs(compute_loss(w1, X1, y1,f) - compute_loss(w2, X2, y2,f)).data.numpy()
        param_distance[epoch] = torch.norm(w1 - w2).data.numpy()
    return loss_distance, param_distance


'''
Runs momentumSGD to comput the needed stability metrics. It computes the Euclidean parametric
distance and the difference between losses at each epoch.

Input: X - feature matrix (numpy array)
       y - output vetor (numpy array)
       f - loss function
       epochs - number of epochs to run on
       alpha - learning rate
       beta - momentum averaging rate
'''
def msgd_stability(X, y, f, epochs, alpha, beta):
    ## Create the differring datasets for each run.
    X1, y1, X2, y2  = data_split(X,y)
    n, d = X1.shape
    X1, y1, X2, y2 = Variable(torch.Tensor(X1), requires_grad=True), Variable(torch.Tensor(y1), requires_grad=True), Variable(torch.Tensor(X2), requires_grad=True), Variable(torch.Tensor(y2), requires_grad=True)
    w1 = Variable(torch.Tensor(initialize(d)), requires_grad=True)
    w2, w1_prev, w2_prev = w1, w1, w1
    loss_distance = [0 for _ in range(epochs)]
    param_distance = [0 for _ in range(epochs)] ## We use L2-norm here
    loss_distance[0] = torch.abs(compute_loss(w1, X1, y1, f) - compute_loss(w2, X2, y2, f)).data.numpy()
    param_distance[0] = torch.norm(w1 - w2).data.numpy()
    for epoch in tqdm(range(1, epochs)):
        for _ in range(n):
            index = np.random.randint(0,n)
            new_w1 = descent.msgd_step(w1, w1_prev, f, X1[index], y1[index], alpha, beta)
            new_w2 = descent.msgd_step(w2, w2_prev, f, X2[index], y2[index], alpha, beta)
            w1_prev, w2_prev = w1, w2
            w1, w2 = new_w1, new_w2
        loss_distance[epoch] = torch.abs(compute_loss(w1, X1, y1, f) - compute_loss(w2, X2, y2, f)).data.numpy()
        param_distance[epoch] = torch.norm(w1 - w2).data.numpy()
    return loss_distance, param_distance



'''
Splits the data into two datasets that differ in one data point.
Input: X - initial data (numpy array)
       y - initial output vector (numpy array)
Ouput: the two differing by 1 datasets X1,y1 and X2,y2
'''
def data_split(X,y):
    assert len(X) == len(y)
    n = len(X)
    X1, y1 = np.delete(X, n-1, 0), np.delete(y, n-1, 0)
    X2, y2 = np.delete(X, n-1, 0), np.delete(y, n-1, 0)
    index = np.random.randint(0, n-1)
    X2[index], y2[index] = X[n-1], y[n-1]
    return X1, y1, X2, y2


'''
Initialize the parameter vector.

Input: d - dimension of the parameter space

Output: a parameter vector
'''
def initialize(d):
    mu, sigma = 0, 10
    return np.random.normal(mu, sigma, d)


'''
Computes the loss on the dataset with the given parameter.

Input: w - parameter vector (torch Variable)
       X - feature matrix (torch Variable)
       y - output vector (torch Variable)
       f - objective loss function

Output: the loss averaged over the dataset
'''
def compute_loss(w, X, y, f):
    n, d = X.shape
    total_loss = 0
    for i in range(n):
        x_i, y_i = X[i], y[i]
        total_loss += f(w, x_i, y_i)
    total_loss /= float(n)
    return total_loss


if __name__=="__main__":
    alpha = 0.01
    beta = 0.01
    epochs = 100
    f = square_loss
    X,y = lin_gen(n_samples=1001, n_features=100, n_informative=15, bias=3.0, noise=1.0)
    # loss_distance, param_distance = sgd_stability(X, y, f, epochs, alpha)
    loss_distance, param_distance = msgd_stability(X, y, f, epochs, alpha, beta)
    epochs = [i for i in range(1,epochs + 1)]
    plt.plot(epochs, param_distance)
    plt.show()
