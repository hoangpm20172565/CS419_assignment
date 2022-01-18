'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''
import numpy as np


class LogisticRegression:

    def __init__(self, alpha=0.01, regLambda=0.01, epsilon=0.0001, maxNumIters=10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.JHist = None
        self.theta = None

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        h = self.sigmoid(X.dot(theta))
        loss = np.where(y, -np.log(h), -np.log(1-h))
        regularizer = regLambda*np.sum(theta ** 2)/2
        # print(f'loss: {loss}, regularizer: {regularizer}')
        return np.asscalar(np.sum(loss) + regularizer)

    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        gradient = X.T.dot(self.sigmoid(X.dot(theta)) - y) + regLambda*theta
        gradient[0] = np.sum(self.sigmoid(X.dot(theta)) - y)
        return gradient

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        # add 1s column
        n, d = X.shape
        X = np.c_[np.ones((n, 1)), X]
        if not self.theta:
            self.theta = np.zeros(shape=(d+1, 1))

        print(self.theta)

        self.JHist = []
        for i in range(self.maxNumIters):
            self.JHist.append(
                (self.computeCost(self.theta, X, y, self.regLambda), self.theta))
            if (i+1)%500==0:
                print("Iteration: ", i + 1, " Cost: ",
                      self.JHist[i][0], " Theta: ", self.theta)
            # update theta
            old_theta = self.theta
            self.theta = self.theta - self.alpha * \
                self.computeGradient(self.theta, X, y, self.regLambda)
            # checking the converged condition
            if np.asscalar(np.sqrt(np.sum((old_theta-self.theta)**2))) <= self.epsilon:
                print(f'Theta has converged at epoch {i}')
                break

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        X = np.c_[np.ones((len(X), 1)), X]
        return self.sigmoid(X.dot(self.theta))

    def sigmoid(self, Z):
        '''
        Computes the sigmoid function 1/(1+exp(-z))
        '''
        return 1/(1 + np.exp(-Z))
