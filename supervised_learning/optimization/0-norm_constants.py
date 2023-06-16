#!usr/bin/env python3
"""placeholder"""
import numpy as np


def normalization_constants(X):
    """calculate the normalization (standardization) constant of matrix
    X: numpy.ndarray of shape (m, nx) to normalize
    m: number of data points
    nx: number of features
    Returns the mean and standard deviation of each feature
    """
    mean = np.mean(X, axis=0)
    standard_deviation = np.std(X, axis=0)

    return mean, standard_deviation
