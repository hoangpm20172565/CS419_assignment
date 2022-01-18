'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8):
        '''
        Constructor
        '''
        #TODO
        # construct a regularizer matrix
        self.degree = degree
        self.Lambda = regLambda


    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not inlude the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        '''
        #TODO
        poly_arr = np.expand_dims(X, axis=1)
        X = poly_arr
        for i in range(2, degree+1):
            temp = poly_arr
            poly_arr = np.append(poly_arr, X**i, axis=1)
            del temp

        return poly_arr
        

    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        '''
        #TODO
        n = len(X)
        X = self.polyfeatures(X, self.degree)

        # scaling
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X = (X-self.mean)/self.std

        # add 1s column
        Xex = np.c_[np.ones([n, 1]), X];

        n, d = Xex.shape
        d = d - 1  # remove 1 for the extra column of ones we added to get the original num features

        # construct reg matrix
        regMatrix = self.Lambda * np.eye(d + 1)
        regMatrix[0, 0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.theta = np.linalg.pinv(Xex.T.dot(Xex) + regMatrix).dot(Xex.T).dot(y);
        
        
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        '''
        # TODO
        X = self.polyfeatures(X, self.degree)
        X = (X-self.mean)/self.std
        Xex = np.c_[np.ones([len(X), 1]), X]
        return Xex.dot(self.theta)



#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------


def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree
        
    Returns:
        errorTrains -- errorTrains[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTests -- errorTrains[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]
        
    Note:
        errorTrains[0:1] and errorTests[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''
    
    n = len(Xtrain);
    
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
    for i in range(2, n):
        Xtrain_subset = Xtrain[:(i+1)]
        Ytrain_subset = Ytrain[:(i+1)]
        model = PolynomialRegression(degree, regLambda)
        model.fit(Xtrain_subset,Ytrain_subset)
        
        predictTrain = model.predict(Xtrain_subset)
        err = predictTrain - Ytrain_subset;
        errorTrain[i] = np.multiply(err, err).mean();
        
        predictTest = model.predict(Xtest)
        err = predictTest - Ytest;
        errorTest[i] = np.multiply(err, err).mean();
    
    return (errorTrain, errorTest)