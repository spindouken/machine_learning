#!/usr/bin/env python3
"""
calculates the mean and covariance of a data set
"""
import numpy as np


def mean_cov(X):
    """
    X is a numpy.ndarray of shape (n, d)
        containing the data set:
            n: the number of data points
            d: the number of dimensions in each data point

    If X is not a 2D numpy.ndarray,
        raise a TypeError with the message:
        'X must be a 2D numpy.ndarray'
    If n is less than 2,
        raise a ValueError with the message:
            'X must contain multiple data points'

    Returns: mean, cov:
        mean is a numpy.ndarray of shape (1, d)
            containing the mean of the data set
        cov is a numpy.ndarray of shape (d, d)
            containing the covariance matrix of the data set

    formula for calculating the covariance matrix (cov):

    cov = (Xcentered.T @ Xcentered) / (n - 1)

    formula explained:
        Xcentered: data matrix after centering it
            around zero by subtracting the mean
            from each data point. Shape = (n, d)
        Xcentered.T: Transpose of Xcentered, changing its shape to (d, n)
        n: number of data points
        d: number of dimensions in each data point
        @: matrix multiplication
        (n - 1): used for the unbiased estimator
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    # Get the number of data points (n) and dimensions (d)
    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    # calculate mean for each feature
    #   sum along each feature and divide by total number of data points
    mean = np.sum(X, axis=0) / n
    mean = mean[:, np.newaxis]  # convert to shape (d, 1) for multinomral class

    # center data by subtracting the mean from each feature
    Xcentered = X - mean.T

    # calculate the covariance matrix using matrix multiplication
    # -----------------------------------------------------------
    # multiplication of Xcentered.T shape (d, n) and Xcentered shape (n, d)
    #   gives shape of (d, d), the shape of the covariance matrix
    # divide by (n - 1) for the unbiased estimator
    covarianceMatrix = (Xcentered.T @ Xcentered) / (n - 1)

    return mean, covarianceMatrix
