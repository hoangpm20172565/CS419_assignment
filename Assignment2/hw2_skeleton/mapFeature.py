import numpy as np

def mapFeature(x1, x2):
    '''
    Maps the two input features to quadratic features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the 6th power polynomial
        
    Arguments:
        X1 is an n-by-1 column matrix
        X2 is an n-by-1 column matrix
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    '''
    n = len(x1)
    new_feature = np.ones((n,28))
    new_feature[:,0] = x1
    new_feature[:,1] = x2

    k = 2
    for i in range(2,7):
        for j in range(i+1):
            new_feature[:,k] = (x1**j)*(x2**(i-j))
            k += 1
    return new_feature

