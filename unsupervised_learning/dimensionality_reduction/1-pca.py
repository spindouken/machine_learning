#!/usr/bin/env python3
"""
performs PCA on a dataset
"""
import numpy as np


def pca(X, ndim):
    """
    X is a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
    ndim is the new dimensionality of the transformed X
    Returns: T, a numpy.ndarray of shape (n, ndim) containing the transformed
        version of X
    """
    # singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # select the top 'ndim' right singular vectors
    #   these vectors form our weights matrix W
    W = Vt[:ndim].T

    # transform the original dataset
    #   perform matrix multiplication between X and W to get T
    T = np.dot(X, W)

    return T
