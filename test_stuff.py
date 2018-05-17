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
# visualize_y_locations(noise=1.0)


dists_sgd = torch.load('final_result_dist.pt')
dists_0_5msgd = torch.load('final_result_dist_msgd.pt')
dists_0_3msgd = torch.load('final_result_dist_0.3msgd.pt')
dists_0_7msgd = torch.load('final_result_dist_0.7msgd.pt')

epochs = [i for i in range(len(dists_sgd))]

plt.plot(epochs, dists_sgd, color='r', linestyle='--', label='SGD')
plt.plot(epochs, dists_0_3msgd, color=(0,0.5,0.8), linestyle='--', label='MSGD 0.3')
plt.plot(epochs, dists_0_5msgd, color=(0,0.3,0.9), linestyle='--', label='MSGD 0.5')
plt.plot(epochs, dists_0_7msgd, color=(0,0,1), linestyle='--', label='MSGD 0.7')
plt.xlabel('Epochs')
plt.ylabel('Normalized Euclidean distance b/w parameters')
plt.legend()
plt.show()



long_dists_sgd = torch.load('final_result_dist_0.0msgd_500epochs.pt')
long_dists_0_5sgd = torch.load('final_result_dist_0.5msgd_500epochs.pt')

epochs = [i for i in range(len(long_dists_sgd))]
plt.plot(epochs, long_dists_sgd, color='r', linestyle='--', label='SGD')
plt.plot(epochs, long_dists_0_5sgd, color=(0,0.5,0.8), linestyle='--', label='MSGD 0.5')
plt.xlabel('Epochs')
plt.ylabel('Normalized Euclidean distance b/w parameters')
plt.show()


long_errors_sgd = torch.load('final_result_error_0.0msgd_500epochs.pt').numpy()

plt.plot(epochs, long_errors_sgd /100)
plt.show()
