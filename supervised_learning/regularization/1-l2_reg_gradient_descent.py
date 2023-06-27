#!/usr/bin/env python3
"""
updates the weight and biases of a neural network
using gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Y: one-hot numpy.ndarray of shape (classes, m)
        contains the correct labels for the data
        classes: number of classes
        m: number of data points
    weights: dictionary of the weights and biases of the neural network
    cache: dictionary of the outputs of each layer of the neural network
    alpha: learning rate
    lambtha: the L2 regularization parameter
    L: number of layers of the network
    Notes: network uses tanh activations on each layer except the last
        ...last layer uses a softmax activation
    Updates the weights and biases of the network using gradient descent
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for layerIndex in range(L, 0, -1):
        activation_of_prev_layer = cache['A' + str(layerIndex - 1)]
        dW = np.matmul(
            dZ, activation_of_prev_layer.T
            ) / m + (lambtha / m) * weights['W' + str(layerIndex)]
        db = np.sum(dZ, axis=1, keepdims=True) / m
        if layerIndex > 1:
            dZ = np.matmul(
                weights['W' + str(layerIndex)].T, dZ
                ) * (1 - activation_of_prev_layer ** 2)
        weights['W' + str(layerIndex)] -= alpha * dW
        weights['b' + str(layerIndex)] -= alpha * db
