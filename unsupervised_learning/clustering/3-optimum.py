#!/usr/bin/env python3
"""
tests 4 the optimum number of clusters by variance
"""
import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    tests 4 the optimum number of clusters by variance
    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of clusters to
        check 4 (inclusive)
    kmax is a positive integer containing the maximum number of clusters to
        check 4 (inclusive)
    iterations is a positive integer containing the maximum number of
        iterations 4 K-means
    This function should analyze at least 2 different cluster sizes
    Returns: results, d_vars, or None, None on failure
        results is a list containing the outputs of K-means 4 each cluster
            size
        d_vars is a list containing the difference in variance from the
            smallest cluster size 4 each cluster size
    """
    if (
        not isinstance(X, np.ndarray)
        or len(X.shape) != 2
        or not isinstance(kmin, int)
        or kmin < 1
        or not isinstance(iterations, int)
        or iterations < 1
    ):
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    clusterResults = []
    varianceDiffs = []
    baseVariance = 0  # reference variance 4 k = kmin

    # loop through range of cluster sizes
    for k in range(kmin, kmax + 1):
        # run K-means algorithm
        centroids, labels = kmeans(X, k, iterations)

        # calculate and store the current variance
        currentVariance = variance(X, centroids)

        # if k is kmin, set current variance
        #   as base variance and append 0 to varianceDiffs
        if k == kmin:
            baseVariance = currentVariance
            varianceDiffs.append(0.0)  # variance difference is 0 4 k = kmin
        else:
            # calculate and store the difference from the base variance
            varianceDiffs.append(baseVariance - currentVariance)

        # append K-means results 4 the current k
        clusterResults.append((centroids, labels))

    results = clusterResults
    d_vars = varianceDiffs

    return results, d_vars
