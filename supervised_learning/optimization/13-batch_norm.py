#!/usr/bin/env python3
"""
normalizes an unactivated output of a neural network using batch normalization
"""


def batch_norm(Z, gamma, beta, epsilon):
    """
    Z: numpy.ndarray of shape (m, n) that should be normalized
        m: number of data points
        n: number of features in Z
    gamma: numpy.ndarray of shape (1, n) containing the scales used for
        batch normalization
    beta: numpy.ndarray of shape (1, n) containing the offsets used for
        batch normalization
    epsilon: small number used to avoid division by zero
    Returns the normalized Z matrix
    """
    mean = Z.mean(axis=0)
    variance = Z.var(axis=0)
    Znorm = (Z - mean) / ((variance + epsilon) ** 0.5)
    Znormalized_scaled = gamma * Znorm + beta
    return Znormalized_scaled
