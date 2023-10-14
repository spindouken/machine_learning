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

    def predict(self, X_s):
        """
        predicts the mean and standard deviation of points
            in a Gaussian process
        X_s is a numpy.ndarray of shape (s, 1) containing all of
            the points whose mean and standard deviation should be calculated
            s is the number of sample points
        Returns: mu, sigma
            mu is a numpy.ndarray of shape (s,) containing the mean
                for each point in X_s, respectively
            sigma is a numpy.ndarray of shape (s,) containing the standard
                deviation for each point in X_s, respectively
        """
        # calculate covariance matrix between training data and sample points
        # (used to compute relationship betwn new sample points&existing data)
        K_s = self.kernel(self.X, X_s)

        # calculate the inverse of the training data covariance matrix
        # (required to find the mean and covariance of predictions)
        K_inv = np.linalg.inv(self.K)

        # calculate covariance matrix for sample points K_ss
        # (required to account for inherent variability in prediction)
        K_ss = self.kernel(X_s, X_s)

        # compute intermediate mean vector using formula:
        #   mu_s = K_s^T * K_inv * Y
        # (computes predicted mean values for new points)
        mu = K_s.T @ K_inv @ self.Y

        # reshape mean vector to match output shape
        mu = mu.reshape(-1)

        # compute covariance matrix for sample points using formula:
        #   cov_s = K_ss - K_s^T * K_inv * K_s
        # (required for capturing how predictions vary around the mean)
        cov_s = K_ss - (K_s.T @ K_inv @ K_s)

        # extract variances (diagonal of the covariance matrix)
        # (essential for providing standard deviation of predictions)
        sigma = np.diag(cov_s)
        return mu, sigma
