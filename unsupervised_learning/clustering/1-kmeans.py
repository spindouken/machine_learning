#!/usr/bin/env python3
"""
performs K-means on a dataset
"""
import numpy as np


def initialize(X, k):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset that will be
        used for K-means clustering
            n is the number of data points
            d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    Returns: a numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or not isinstance(k, int):
        return None
    if len(X.shape) != 2 or k < 1:
        return None

    n, d = X.shape

    # retrieve minimum and maximum values for each dimension of each data point
    minValues = np.min(X, axis=0)
    maxValues = np.max(X, axis=0)

    # multivariate uniform distribution
    centroids = np.random.uniform(minValues, maxValues, (k, d))

    # return initialized centroids
    return centroids


def kmeans(X, k, iterations=1000):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of
        iterations that should be performed
    If no change in the cluster centroids occurs between iterations,
        function should return
    Initialize the cluster centroids using a multivariate uniform distribution
        (based on 0-initialize.py)
    If a cluster contains no data points during the update step, reinitialize
        its centroid
    Returns: C, clss, or None, None on failure
    C is a numpy.ndarray of shape (k, d) containing the centroid means for
            each cluster
        clss is a numpy.ndarray of shape (n,) containing
            the index of the cluster
                in C that each data point belongs to
    """
    clusterCentroids = initialize(X, k)

    # retrieve minimum and maximum values for each dimension of each data point
    minValues = np.min(X, axis=0)
    maxValues = np.max(X, axis=0)

    n, d = X.shape

    # iterate through assigned maximum number of iterations
    for i in range(iterations):
        dataDistances = np.zeros((n, k))

        # calculate distances between each data point and cluster centroid
        #   and store in dataDistances matrix
        for x in range(n):
            for y in range(k):
                dataDistances[x, y] = np.linalg.norm(
                    X[x] - clusterCentroids[y]
                    )
        # holds index of cluster centroid with minimum distance to each point
        clss = np.argmin(dataDistances, axis=1)
        # make a copy of clusterCentroids to check for convergence later
        initialCentroids = clusterCentroids.copy()

        # iterate through each cluster
        for x in range(k):
            # retrieve all data points in current cluster
            clusterPoints = X[clss == x]
            # if cluster contains no data points, reinitialize cluster centroid
            if len(clusterPoints) == 0:
                clusterCentroids[x] = np.random.uniform(
                    minValues, maxValues, (1, d)
                )
            else:
                # otherwise, update cluster centroid to mean of all data points
                clusterCentroids[x] = np.mean(clusterPoints, axis=0)

        # cconvergence occurs when clusterCentroids no longer change
        #   between iterations, so we return the current clusterCentroids
        if np.all(initialCentroids == clusterCentroids):
            return clusterCentroids, clss

    return clusterCentroids, clss
