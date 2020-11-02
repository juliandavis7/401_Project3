import pandas as pd
import numpy as np

#ACTIVATION FUNCTIONS
def inplace_relu(X):
        """Compute the rectified linear unit function inplace.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        """
        np.maximum(X, 0, out=X)
        
ACTIVATIONS = {'relu': inplace_relu}

# DERIVATIVE OF ACTIVATION FUNCTIONS
def inplace_relu_derivative(Z, J):
    """Apply the derivative of the relu function.
    It exploits the fact that the derivative is a simple function of the output
    value from rectified linear units activation function.
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the rectified linear units activation
        function during the forward pass.
    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    J[Z == 0] = 0
    
DERIVATIVES = {'relu': inplace_relu_derivative}

# LOSS FUNCTIONS
def squared_loss(y_true, y_pred):
        """Compute the squared loss for regression.
        Parameters
        ----------
        y_true : array-like or label indicator matrix
            Ground truth (correct) values.
        y_pred : array-like or label indicator matrix
            Predicted values, as returned by a regression estimator.
        Returns
        -------
        loss : float
            The degree to which the samples are correctly predicted.
        """
        return ((y_true - y_pred) ** 2).mean() / 2
    
# DERIVATIVE OF LOSS FUNCTIONS
def squared_loss_derivative(y_pred, y_true, batch_size):
    return (y_pred - y_true)/batch_size

