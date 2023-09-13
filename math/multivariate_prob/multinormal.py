#!/usr/bin/env python3
"""
represents a Multivariate Normal distribution
"""
import numpy as np


def mean_cov(X):
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate the mean for each dimension
    mean = np.mean(X, axis=0, keepdims=True).T  # Transpose to get shape (d, 1)

    # Initialize the covariance matrix with zeros
    cov = np.zeros((d, d))

    # Loop through each pair of features (i, j)
    for i in range(d):
        for j in range(d):
            # Calculate the covariance between feature i and feature j
            cov[i, j] = np.sum(
                (X[:, i] - mean[i, 0]) * (X[:, j] - mean[j, 0])
            ) / (n - 1)

    return mean, cov


class MultiNormal:
    def __init__(self, data):
        """
        data is a numpy.ndarray of shape (d, n) containing the data set:
            n: the number of data points
            d: the number of dimensions in each data point
        If data is not a 2D numpy.ndarray, raise a TypeError with the message:
            'data must be a 2D numpy.ndarray'
        If n is less than 2, raise a ValueError with the message:
            'data must contain multiple data points'
        Sets public instance variables:
            mean - a numpy.ndarray of shape (d, 1) containing the mean of data
            cov - a numpy.ndarray of shape (d, d)
                containing the covariance matrix data
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')

        n, d = data.shape
        if n < 2:
            raise ValueError('data must contain multiple data points')

        mean, cov = mean_cov(data)

        self.mean = mean
        self.cov = cov
