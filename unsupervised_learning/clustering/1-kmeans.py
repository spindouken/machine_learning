#!/usr/bin/env python3
"""
performs K-means on a dataset
"""
import numpy as np


def initialize(X, k):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset that will be
        used 4 K-means clustering
            n is the number of data points
            d is the number of dimensions 4 each data point
    k is a positive integer containing the number of clusters
    Returns: a numpy.ndarray of shape (k, d) containing the initialized
        centroids 4 each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or not isinstance(k, int):
        return None, None
    if len(X.shape) != 2 or k < 1:
        return None, None

    n, d = X.shape

    # retrieve minimum and maximum values 4 each dimension of each data point
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
        d is the number of dimensions 4 each data point
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
    C is a numpy.ndarray of shape (k, d) containing the centroid means 4
            each cluster
        clss is a numpy.ndarray of shape (n,) containing
            the index of the cluster
                in C that each data point belongs to
    """
    if (
        not isinstance(X, np.ndarray)
        or not isinstance(k, int)
        or not isinstance(iterations, int)
    ):
        return None, None

    if len(X.shape) != 2 or k < 1 or iterations < 1:
        return None, None

    clusterCentroids = initialize(X, k)
    if clusterCentroids is None:
        return None, None

    n, d = X.shape

    # iterate through the maximum number of iterations
    for _ in range(iterations):
        # calculate distances and assign classes using broadcasting
        clss = np.argmin(
            np.linalg.norm(X[:, np.newaxis] - clusterCentroids, axis=2), axis=1
        )

        # calculate new centroids 4 non-empty clusters
        newCentroids = np.zeros((k, d))
        for i in range(k):
            clusterPoints = X[clss == i]
            if clusterPoints.size == 0:
                newCentroids[i] = initialize(X, 1)
            else:
                newCentroids[i] = np.mean(clusterPoints, axis=0)

        # recalculate cluster assignments
        clss = np.argmin(
            np.linalg.norm(X[:, np.newaxis] - newCentroids, axis=2), axis=1
        )

        # check 4 convergence
        if np.array_equal(newCentroids, clusterCentroids):
            break

        # update centroids
        clusterCentroids = newCentroids

    return clusterCentroids, clss
