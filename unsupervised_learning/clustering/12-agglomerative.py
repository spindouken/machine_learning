#!/usr/bin/env python3
"""
performs agglomerative clustering on a dataset
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset
    dist is the maximum cophenetic distance for all clusters

    Performs agglomerative clustering with Ward linkage
    Displays the dendrogram with each cluster displayed in a different color

    Returns: clss, a numpy.ndarray of shape (n,) containing the cluster
        indices for each data point
    """
    # calculate linkage matrix using ward linkage
    linkageMatrix = scipy.cluster.hierarchy.linkage(X, 'ward')
    # generate cluster labels for data points
    clss = scipy.cluster.hierarchy.fcluster(linkageMatrix, dist, 'distance')
    # create dendrogram
    scipy.cluster.hierarchy.dendrogram(linkageMatrix, color_threshold=dist)
    plt.show()

    return clss
