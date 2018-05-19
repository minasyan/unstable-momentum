import torch
import numpy as np
from matplotlib import pyplot as plt

'''
Function used to visualize the non-convex function used in our empirical evaluation:

y = (3/2)*(x)**2 + np.exp(0.6-1 / (100*(x-1)**2)) - 1.5

Inputs: n_samples - number of the samples to generate
        n_features - dimensionality of the data to be generated
        noise - standard deviation of the noise injected in the data

'''
def visualize_y_locations(n_samples=10000, n_features=1, noise=1.0):
    np.random.seed(0)
    w = np.random.normal(loc=0, scale=0.25, size=n_features)
    X = np.random.normal(loc=0, scale=0.25, size=(n_samples, n_features))
    noise = np.random.normal(0, noise, n_samples)

    ## Look at what the function looks like.
    x = np.arange(-1.5,1.5,0.01)
    y = (3/2)*(x)**2 + np.exp(0.6-1 / (100*(x-1)**2)) - 1.5
    plt.xlabel("x = dot(w,z)")
    plt.ylabel("f(x)")
    plt.plot(x,y, 'k--')
    plt.show()

visualize_y_locations(noise=1.0)
