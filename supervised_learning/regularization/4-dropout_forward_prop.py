#!/usr/bin/env python3
"""
conducts forward propagation using Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    X: numpy.ndarray of shape (nx, m) containing the input data for the network
        nx: number of input features
        m: number of data points
    weights: dictionary of the weights and biases of the neural network
    L: number of layers in the network
    keep_prob: the probability that a node will be kept
    Returns a dictionary containing the outputs of each layer and the dropout
        mask used on each layer
    """
    cache = {}
    cache["A0"] = X

    for i in range(1, L + 1):
        Z = np.matmul(
            weights["W" + str(i)], cache["A" + str(i - 1)]
            ) + weights["b" + str(i)]
        if i != L:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            cache["D" + str(i)] = D
            A *= D
            A /= keep_prob
        else:
            A = softmax(Z)
        cache["A" + str(i)] = A

    return cache
