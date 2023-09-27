#!/usr/bin/env python3
"""
performs PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """
    X is a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
            all dimensions have a mean of 0 across all data points
    var is the fraction of the variance that the PCA transformation should
        maintain
    
    variables:
    U, S, Vt: result of SVD on centeredData
        S' contains the singular values, indicating
            the importance of corresponding vectors in Vt
    totalVariance: sum of the squares of all singular values
        represents total data variance
    retainedVariance: cccumulates the variance
        as we consider each singular value in 'S'
    numComponents: Counts the number of singular values
        (principal components) needed to achieve the desired 'var'
        Used to construct 'W'

    result of singular value decomposition (SVD) on mean centered data:
        U: left singular vectors
        S: singular values
        Vt: right singular vectors

    Returns: the weights matrix, W, that maintains var fraction of X‘s
        original variance
        W is a numpy.ndarray of shape (d, nd)
            where nd is the new dimensionality of the transformed X
    """
    # mean centering the data for PCA
    # first center the dataset around the origin
    #   by subtracting the mean of each feature from all data points
    featureMeans = np.mean(X, axis=0)
    B = X - featureMeans

    # singular value decomposition (SVD) on mean centered data
    U, S, Vt = np.linalg.svd(B, full_matrices=False)

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
