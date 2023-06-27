#!/usr/bin/env python3
"""
updates the weights of a neural network with
    Dropout regularization using gradient descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Y: one-hot numpy.ndarray of shape (classes, m) that contains data labels
        classes: number of classes
        m: number of data points
    weights: dictionary of the weights and biases of the neural network
    cache: dictionary of the outputs and dropout masks of each layer of the nn
    alpha: learning rate
    keep_prob: probability that a node will be kept
    L: number of layers of the network
    Weights of the network updated in place
    """
    m = Y.shape[1]
    dZ = (cache["A{}".format(L)] - Y)
    for layer in range(L, 0, -1):
        a = cache["A{}".format(layer - 1)]
        dW = np.matmul(dZ, cache["A{}".format(layer-1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if layer > 1:
            dZ_tmp = (1 - np.square(a))
            mask = cache["D{}".format(layer-1)]
            dZ = np.matmul(
                weights["W{}".format(layer)].T, dZ
                ) * dZ_tmp * mask
            dZ /= keep_prob

        weights["W{}".format(layer)] -= (alpha * dW)
        weights["b{}".format(layer)] -= (alpha * db)
