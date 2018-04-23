import torch

'''
Computes the square loss on a data point.

Input: w - paramater vector
       x - training point
       y - label

Output: loss on the given point.
'''
def square_loss(w, x, y):
    return (y - torch.dot(w,x))**2
