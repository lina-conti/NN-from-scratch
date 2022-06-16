#utility functions for NNClasses
import numpy as np

class Activation(object):
    '''an abstract activation functor class'''
    def __init__(self):
        pass
    def __call__(self, X):
        pass
    def deriv(self, X):
        pass

class ReluActivation(Activation):
    '''Rectified Linear unit: returns x if x > 0, otherwise returns x'''
    def __call__(self, X):
        return np.maximum(0, X)

    def deriv(self, X):
        X = np.where(X <= 0, 0, X)
        X = np.where(X > 0, 1, X)
        return X

class TanhActivation(Activation):
    '''Hyperbolic tangent: maps all values to between -1 and 1'''
    def __call__(self, X):
        return np.tanh(X)

    def deriv(self, X):
        return 1-np.tanh(X)**2

class SigmoidActivation(Activation):
    '''sigmod activation: maps all values to between 0 and 1'''
    def __call__(self, X):
        return 1.0/(1.0+np.exp(-X))

    def deriv(self, X):
        temp = self(X)
        return temp*(1-temp)

# AUXILIARY FUNCTION FOR MLP class
def get_one_hot_batch(batch_y, vector_size = None):
    '''
    batch_y: ndarray of size batch_size with the index of the gold class for each example in the batch
    Output: a batch of one-hot vectors with 1 at the y component for each example
    '''
    #print('shape ', batch_y.shape, 'size, ', batch_y.size)
    one_hot = np.zeros((batch_y.size, (vector_size if vector_size else batch_y.max()+1)))
    # an array of size batch_size with values from 0 to the bacth_size
    rows = np.arange(batch_y.size)
    # set the components at the index of the gold classes to 1
    one_hot[rows, batch_y] = 1
    return one_hot
