import torch

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
