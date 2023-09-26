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


def updateCentroids(X, clss, k, minValues, maxValues):
    """
    update centroids based on the points in each cluster
    if a cluster has no points, reinitialize its centroid
    """
    d = X.shape[1]
    newCentroids = np.zeros((k, d))

    # iterate through each cluster centroid and update
    for x in range(k):
        # retrieve all data points in the current cluster
        clusterPoints = X[clss == x]
        # if cluster contains no data points, reinitialize cluster centroid
        if len(clusterPoints) == 0:
            newCentroids[x] = np.random.uniform(minValues, maxValues, d)
        else:
            # otherwise, update cluster centroid to mean of all data points
            newCentroids[x] = np.mean(clusterPoints, axis=0)
    return newCentroids


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

    # iterate through the maximum number of iterations
    for _ in range(iterations):
        # calculate distances and assign classes using broadcasting
        clss = np.argmin(
            np.linalg.norm(X[:, np.newaxis] - clusterCentroids, axis=2), axis=1
        )

        # make a copy to check for convergence later
        initialCentroids = clusterCentroids.copy()

        # update centroids
        clusterCentroids = updateCentroids(X, clss, k, minValues, maxValues)

        # break if convergence is reached
        if np.all(initialCentroids == clusterCentroids):
            break

    return clusterCentroids, clss
