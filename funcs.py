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
        
def inplace_leaky_relu(X, a = 0.1):
        """Compute the leaky rectified linear unit function inplace.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        a : non-zero coefficient
        """
        Y = X.copy()
        X *= 0
        X += np.where(Y > 0, Y, Y * a)

def inplace_sigmoid(X):
        """Compute the sigmoid function inplace.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        """       
       
        Y = X.copy()
        X *= 0
        X += 1 / (1 + np.exp(1) ** -Y)
        
def inplace_tanh(X):
    """Compute the sigmoid function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    np.tanh(X, out=X)
    
        
ACTIVATIONS = {'relu': inplace_relu,
               'leaky_relu': inplace_leaky_relu,
               'sigmoid': inplace_sigmoid,
               'tanh': inplace_tanh}

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

# DERIVATIVE OF ACTIVATION FUNCTIONS
def inplace_leaky_relu_derivative(Z, J, a):
    """Apply the derivative of the relu function.
    It exploits the fact that the derivative is a simple function of the output
    value from leaky rectified linear units activation function.
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the rectified linear units activation
        function during the forward pass.
    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    J[Z <= 0] = a
    
# DERIVATIVE OF ACTIVATION FUNCTIONS
def inplace_sigmoid_derivative(Z, J):
    """Apply the derivative of the relu function.
    It exploits the fact that the derivative is a simple function of the output
    value from the sigmoid function.
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the rectified linear units activation
        function during the forward pass.
    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    J *= (1 - Z)
    
# DERIVATIVE OF ACTIVATION FUNCTIONS
def inplace_tanh_derivative(Z, J):
    """Apply the derivative of the relu function.
    It exploits the fact that the derivative is a simple function of the output
    value from hyperbolic tangent.
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the rectified linear units activation
        function during the forward pass.
    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    J *= (1 - Z ** 2)
    
DERIVATIVES = {'relu': inplace_relu_derivative,
              'leaky_relu': inplace_leaky_relu_derivative,
              'sigmoid': inplace_sigmoid_derivative,
              'tanh': inplace_tanh_derivative}

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

