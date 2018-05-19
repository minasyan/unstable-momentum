import torch
from data_generation import non_convex_fn
'''
Computes the square loss on a data point
in a standard linear regression setting.

Input: w - paramater vector
       x - training point
       y - label

Output: loss on the given point.
'''
def square_loss(w, x, y):
    return (y - torch.dot(w,x))**2

'''
Computes the square loss on a data point in a
regression setting with a sigmoid activation function.

Input: w - paramater vector
       x - training point
       y - label

Output: loss on the given point.
'''
def square_sigmoid_loss(w, x, y):
    return (y - torch.sigmoid(torch.dot(w,x)))**2


'''
Computes the square loss on a data point in a
regression setting with the non_convex activation
function described in the data_generation file.

Input: w - paramater vector
       x - training point
       y - label

Output: loss on the given point.
'''
def nonconvex_loss(w, x, y):
    ## Has to match what is in data generation.
    dot = torch.dot(w,x)
    inter = (3/2) * ((dot)**2) + torch.exp(0.5 - 1 / (100 * (dot - 1)**2)) - 1.0
    return inter

'''
Simple loss as a dot product of w and x.
'''
def convex_loss(w, x, y):
    ## Simply minimize the dot product
    return torch.dot(w, x)
