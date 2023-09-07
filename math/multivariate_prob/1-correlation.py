#!/usr/bin/env python3
"""
compute the the correlation matrix from a given covariance matrix
"""
import numpy as np


def correlation(C):
    """
    C is a numpy.ndarray of shape (d, d)
        containing a covariance matrix
        d: the number of dimensions

    If C is not a numpy.ndarray, raise a TypeError with the message"
        'C must be a numpy.ndarray'
    If C does not have shape (d, d), raise a ValueError with the message:
        'C must be a 2D square matrix'
    Returns a numpy.ndarray of shape (d, d)
        containing the correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Initialize the correlation matrix with zeros
    correlationMatrix = np.zeros(C.shape)

    """
    calculates the correlation matrix from the given covariance matrix, C

    iterates over each element, C[i, j], in the covariance matrix

    For each element, compute the Pearson correlation coefficient using formula
    correlation[i, j] = covariance[i, j] /
                        sqrt(covariance[i, i] * covariance[j, j])

    calculated correlation coefficient is tstored in the corresponding cell of
        pre-initialized correlationMatrix
    """
    # Loop through each pair of features (i, j)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            # Calculate the correlation σij
            correlationMatrix[i, j] = C[i, j] / np.sqrt(C[i, i] * C[j, j])

    return correlationMatrix
