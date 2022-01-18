"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np


_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    return (X1.dot(X2.T)+1)**_polyDegree



def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    pairwise_dist = np.zeros((len(X1),len(X2)))
    print(X1.shape)
    print(X2.shape)

    for i in range(len(X1)):
        pairwise_dist[i,:] = np.sum((X2-X1[i])**2, axis=1)

    return np.exp(-pairwise_dist/(_gaussSigma**2)/2)



def myCosineSimilarityKernel(X1,X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    matrix_mul = X1.dot(X2.T)
    x1_dist = np.reshape(np.sqrt(np.sum(X1**2, axis=1)), (len(X1), 1))
    x2_dist = np.reshape(np.sqrt(np.sum(X2**2, axis=1)), (1, len(X2)))
    dist_mul = 1/(x1_dist.dot(x2_dist))

    return np.multiply(matrix_mul, dist_mul)

