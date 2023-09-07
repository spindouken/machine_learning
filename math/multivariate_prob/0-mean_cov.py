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

    Error handling:
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
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate the mean for each dimension and keep the dimensions
    mean = np.mean(X, axis=0, keepdims=True)

    # Initialize the covariance matrix with zeros
    cov = np.zeros((d, d))

    """
    Calculate the covariance matrix using the sample covariance formula
        σij = 1/(n−1) n∑k=1 (x[ki]−μ[i])(x[kj]−μ[j])

    Steps:
    X[:, i] extracts all data points for feature i
    mean[0, i] gives us μi, the mean of feature i
    X[:, i] - mean[0, i] calculates (xki−μi) for all k
    Similarly, X[:, j] - mean[0, j] calculates (xkj−μj) for all k
    (X[:, i] - mean[0, i]) * (X[:, j] - mean[0, j])
        computes (xki−μi)(xkj−μj) for all k
    np.sum((X[:, i] - mean[0, i]) * (X[:, j] - mean[0, j]))
        sums up all these products
    / (n - 1) divides the sum by n−1 to get the sample covariance σij

    The calculated covariance σij is then stored in cov[i, j]
    """
    # Loop through each pair of features (i, j)
    for i in range(d):
        for j in range(d):
            # Calculate the covariance between feature i and feature j
            cov[i, j] = np.sum(
                (X[:, i] - mean[0, i]) * (X[:, j] - mean[0, j])
            ) / (n - 1)

    return mean, cov
