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
    totalVariance = np.sum(S ** 2)

    retainedVariance = 0
    numComponents = 0

    # loop through each singular value in 'S'
    for singularValue in S:
        retainedVariance += (singularValue ** 2) / totalVariance
        numComponents += 1
        # stop when we hit the required variance
        if retainedVariance >= var:
            break

    # grab the top 'numComponents' columns from
    #   Vt's transpose for our weight matrix W
    # these columns capture the most significant
    #   'directions' of variance in our dataset
    W = Vt.T[:, :numComponents]

    return W
