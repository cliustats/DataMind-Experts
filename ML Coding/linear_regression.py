############################
#### Linear Regression  ####
############################


# simple linear Regression

import numpy as np


class LinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X, y):
        n = len(X)  
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        numerator = 0
        denominator = 0
        for i in range(n):
            numerator += (X[i] - x_mean) * (y[i] - y_mean)
            denominator += (X[i] - x_mean) ** 2
        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.slope * x + self.intercept)
        return y_pred


### Test ###

X = np.array([1,2,3,4,5])
y = np.array([2,4,5,4,5])
lr = LinearRegression()
lr.fit(X, y)
print(lr.slope)
print(lr.intercept)
y_pred = lr.predict(X)
print(y_pred)


### Vectorized  ###
import numpy as np

class LinearRegression:
    def __init__(self):
        self.W = None

    def fit(self, X, y):
        '''
        X: n x d
        '''
        # Add bias term to X --> [1, X]
        n = X.shape[0]
        X = np.hstack((np.ones((n, 1)), X))
        self.W = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        n = X.shape[0]
        X = np.hstack((np.ones((n, 1)), X))
        return X @ self.W


# test
X = np.array([2,2], [4,5], [7,8])
y = np.arrary([7. 17, 26])

# fit linear regression model
lr = LinearRegression()
lr.fit(X, y)
print(lr.W)

# make predictions on new data
X_new = np.array([10, 11], [12, 13])
y_pred = lr.predict(X_new)
print(y_pred)



#######   improvements on simple LR   ########

import numpy as np

class LinearRegression:
    def __init__(self, regul=0):
        self.regul = regul
        self.W = None

    def fit(self, X, y, lr=0.01, num_iter=1000):
        # input validation
        if len(X) != len(y) or len(X) == 0:
            raise ValueError("X and y must have the same length and cannot be empty")

        # add bias term to X -> [1, X]
        X = np.hstack((np.ones(len(X), 1), X))

        # Initialize W to zeros
        self.W = np.zeros(X.shape[1])
