import torch
from torch.autograd import Variable
import numpy as np
import descent
from tqdm import tqdm
from objectives import square_loss, square_sigmoid_loss, nonconvex_loss, convex_loss
from data_generation import lin_gen, pol_gen, fried1_gen, linear_prob_gen, non_convex_prob
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
    X1, y1 = Variable(torch.Tensor(X1)), Variable(torch.Tensor(y1))
    X2, y2 = Variable(torch.Tensor(X2)), Variable(torch.Tensor(y2))
    ## Initialize parameters.
    w1 = Variable(torch.Tensor(initialize(d)), requires_grad=True)
    w2 = w1
    loss_distance = [0 for _ in range(epochs)]
    param_distance = [0 for _ in range(epochs)] ## We use L2-norm here
    loss_distance[0] = float(compute_loss(w2, X2, y2, f).data)
    param_distance[0] = float(torch.norm(w1 - w2).data)
    indices = np.random.permutation(n)
    for epoch in tqdm(range(1, epochs)):
        for i in range(n):
            index = indices[i]
            w1 = descent.sgd_step(w1, f, X1[index], y1[index], alpha)
            w2 = descent.sgd_step(w2, f, X2[index], y2[index], alpha)
        normw1 = torch.div(w1.data, torch.norm(w1.data))
        normw2 = torch.div(w2.data, torch.norm(w2.data))
        loss_distance[epoch] = float(compute_loss(w2, X2, y2,f).data)
        param_distance[epoch] = float(torch.dist(normw1, normw2))
    return loss_distance[5:], param_distance[5:]


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
    X1, y1 = Variable(torch.Tensor(X1)), Variable(torch.Tensor(y1))
    X2, y2 = Variable(torch.Tensor(X2)), Variable(torch.Tensor(y2))
    w1 = Variable(torch.Tensor(initialize(d)), requires_grad=True)
    w2, w1_prev, w2_prev = w1, w1, w1
    loss_distance = [0 for _ in range(epochs)]
    param_distance = [0 for _ in range(epochs)] ## We use L2-norm here
    loss_distance[0] = float(compute_loss(w2, X2, y2, f).data)
    param_distance[0] = float(torch.norm(w1 - w2).data)
    indices = np.random.permutation(n)
    for epoch in tqdm(range(1, epochs)):
        for i in range(n):
            index = indices[i]
            new_w1 = descent.msgd_step(w1, w1_prev, f, X1[index], y1[index], alpha, beta)
            new_w2 = descent.msgd_step(w2, w2_prev, f, X2[index], y2[index], alpha, beta)
            w1_prev, w2_prev = w1, w2
            w1, w2 = new_w1, new_w2
        normw1 = torch.div(w1.data, torch.norm(w1.data))
        normw2 = torch.div(w2.data, torch.norm(w2.data))
        loss_distance[epoch] = float(compute_loss(w2, X2, y2, f).data)
        param_distance[epoch] = float(torch.dist(normw1, normw2))
    return loss_distance[5:], param_distance[5:]



'''
Splits the data into two datasets that differ in one data point.
Input: X - initial data (numpy array)
       y - initial output vector (numpy array)
Ouput: the two differing by 1 datasets X1,y1 and X2,y2
'''
def data_split(X,y):
    assert len(X) == len(y)
    n = len(X)
    print(X[n-1], y[n-1])
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
    # np.random.seed(0)
    mu, sigma = 0, 0.25
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
    alpha = 0.001
    beta = 0.001
    epochs = 200
    nruns = 10
    f = nonconvex_loss
    loss_distance_avg_sgd = np.zeros(epochs)
    loss_distance_avg_msgd = np.zeros(epochs)
    param_distance_avg_sgd = np.zeros(epochs)
    param_distance_avg_msgd = np.zeros(epochs)
    for i in range(nruns):
        # X,y = lin_gen(n_samples=101, n_features=100, n_informative=100, bias=0, noise=10.0)
        X,y = non_convex_prob(n_samples=101, n_features=20, noise=0.3)
        # X,y = fried1_gen(n_samples=1001, n_features=100, noise=10.0)
        loss_distance, param_distance = msgd_stability(X, y, f, epochs, alpha, beta)
        param_distance_avg_msgd += np.array(param_distance)
        loss_distance_avg_sgd += np.array(loss_distance)

        loss_distance, param_distance = sgd_stability(X, y, f, epochs, alpha)
        param_distance_avg_sgd += np.array(param_distance)
        loss_distance_avg_msgd += np.array(loss_distance)


    param_distance_avg_sgd /= float(nruns)
    param_distance_avg_msgd /= float(nruns)
    loss_distance_avg_sgd /= float(nruns)
    loss_distance_avg_msgd /= float(nruns)
    # loss_distance, param_distance = sgd_stability(X, y, f, epochs, alpha, beta)
    param_distance_avg_sgd = np.loadtxt("sgd_param_convex.csv", delimiter=",")
    param_distance_avg_msgd = np.loadtxt("msgd_param_convex.csv", delimiter=",")
    loss_distance_avg_sgd = np.loadtxt("sgd_loss_convex.csv", delimiter=",")
    loss_distance_avg_msgd = np.loadtxt("msgd_loss_convex.csv", delimiter=",")
    epochs = 200
    epochs = [i for i in range(5, epochs)]
    plt.plot(epochs, param_distance_avg_sgd, 'r--', label='SGD')
    plt.plot(epochs, param_distance_avg_msgd, 'b--', label='MSGD')
    plt.xlabel("Number of epochs")
    plt.ylabel("Euclidean distance between parameters")
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel("Normalized euclidean distance btw parameters")
    plt.show()

    plt.plot(epochs, loss_distance_avg_sgd, 'r--', label='SGD')
    plt.plot(epochs, loss_distance_avg_msgd, 'b--', label='MSGD')
    plt.xlabel("Number of epochs")
    plt.ylabel("Objective loss")
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel("Objective loss")
    plt.show()
