import torch
import numpy as np
from matplotlib import pyplot as plt


def visualize_y_locations(n_samples=10000, n_features=1, noise=1.0):
    np.random.seed(0)
    w = np.random.normal(loc=0, scale=0.25, size=n_features)
    X = np.random.normal(loc=0, scale=0.25, size=(n_samples, n_features))
    noise = np.random.normal(0, noise, n_samples)
    y = np.dot(X, w) + noise
    plt.hist(y, bins=100)
    plt.show()
    y = (3/2)*y**2 + np.exp(0.5-1 / (100*(y-1)**2)) - 1.0
    plt.hist(y, bins=100)
    plt.show()

    ## Look at what the function looks like.
    x = np.arange(-1.5,1.5,0.01)
    y = (3/2)*(x)**2 + np.exp(0.5-1 / (100*(x-1)**2)) - 1.0
    plt.plot(x,y)
    plt.show()


    # ## Visualize non convexity.
    # ws = []
    # for _ in range(100):
    #     w = np.random.normal(loc=0, scale=0.25, size=n_features)
    #     ws.append(w)
    # ws.sort()
    # losses = []
    # for w in ws:
    #     dot = np.dot(X,w)




visualize_y_locations(noise=1.0)
