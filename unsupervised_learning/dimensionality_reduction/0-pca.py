#!/usr/bin/env python3
"""
performs PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """
    X is a numpy.ndarray of shape (n, d) where:
    """
    # singular value decomposition (SVD) on mean centered data
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # summing up the squares of singular values to get total variance
    totalVariance = np.sum(S**2)

    # calculate the retained variance for each component
    retainedVariance = np.cumsum(S**2) / totalVariance

    # find the number of components needed to retain the given variance
    numComponents = np.argmax(retainedVariance >= var) + 1

    # grab the top 'numComponents' columns from
    #   Vt's transpose for our weight matrix W
    # these columns capture the most significant
    #   'directions' of variance in our dataset
    W = Vt.T[:, :numComponents]

    return W
