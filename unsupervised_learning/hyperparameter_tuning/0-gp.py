#!/usr/bin/env python3
"""
creates the class GaussianProcess that
    represents a noiseless 1D Gaussian process
"""
import numpy as np


class GaussianProcess:
    """
    represents a noiseless 1D Gaussian process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        X_init is a numpy.ndarray of shape (t, 1) representing
            the inputs already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing
            the outputs of the black-box function for each input in X_init
        t is the number of initial samples
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of
            the black-box function
        Sets the public instance attributes X, Y, l, and sigma_f
            corresponding to the respective constructor inputs
        Sets the public instance attribute K, representing
            the current covariance kernel matrix for the Gaussian process
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        calculates the covariance kernel matrix between two matrices
        X1 is a numpy.ndarray of shape (m, 1)
        X2 is a numpy.ndarray of shape (n, 1)
        the kernel should use the Radial Basis Function (RBF)
        Returns: the covariance kernel matrix as a numpy.ndarray
            of shape (m, n)
        """
        # calculate sum of squares for each row in X1
        #   and reshape to column vector
        X1sum = np.sum(X1**2, 1).reshape(-1, 1)
        # calculate sum of squares for each row in X2
        X2sum = np.sum(X2**2, 1)
        # compute squared Euclidean distance
        #   between each pair of points in X1 and X2
        squaredDistance = X1sum + X2sum - 2 * (X1 @ X2.T)
        # calculate the covariance kernel matrix using the RBF kernel formula
        #   RBF = K(x_i, x_j) = sigma_f^2 * exp(-0.5 / l^2 * squaredDistance)
        covarianceKernelMatrix = (self.sigma_f**2) * np.exp(
            -0.5 / (self.l**2) * squaredDistance
        )
        return covarianceKernelMatrix
