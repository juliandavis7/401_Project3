import numpy as np

class LR:
    
    def fit(self, X_train, y_train):
        # create vector of ones...
        ones = np.ones(shape=len(X_train))[..., None]
        #...and add to feature matrix
        X = np.concatenate((ones, X_train), 1)
        #calculate coefficients using closed-form solution
        self.coeffs = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y_train)
        
    def predict(self, X_test):
        ones = np.ones(shape=len(X_test))[..., None]
        X_test = np.concatenate((ones, X_test), 1)
        y_hat = X_test.dot(self.coeffs)
        return y_hat
