#!/usr/bin/env python3
"""
determines the steady state probabilities of a regular markov chain
"""
import numpy as np


def regular(P):
    """
    P is a is a square 2D numpy.ndarray of shape (n, n) representing the
        transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady state
        probabilities, or None on failure
    """
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    # np.abs(eigenvalues - 1) creates an array containing
    #   the absolute value differences between each eigenvalue and 1
    # np.argmin() then finds the index of the smallest value in that array
    # this is why we use closestToOneIdx
    #   (the index of the eigenvalue closest to 1)
    closestToOneIdx = np.argmin(np.abs(eigenvalues - 1))

    # extract the eigenvector corresponding to the eigenvalue closest to 1
    # we use this index (of a columns' row numbers) to slice
    #   the eigenvectors array and obtain the relevant eigenvector
    eigen_vOne = eigenvectors[:, closestToOneIdx]

    if eigen_vOne.size == 0:
        return None

    # normalize the eigenvector
    #   s=v/(sum(v))
    steadyStateProbs = eigen_vOne / eigen_vOne.sum()
    steadyStateProbs = steadyStateProbs.reshape((1, -1))

    return steadyStateProbs
