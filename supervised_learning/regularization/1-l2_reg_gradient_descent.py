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
    lambtha: L2 regularization parameter
    L: number of layers of the network
    Notes: network uses tanh activations on each layer except the last
        ...last layer uses a softmax activation
    Updates the weights and biases of the network using gradient descent
    """
    m = Y.shape[1]  # Number of data points
    dZ = cache['A' + str(L)] - Y  # Calculate the error derivative

    # Loop over each layer in reverse order (from output to input)
    for layerIndex in range(L, 0, -1):
        A = cache['A' + str(layerIndex - 1)]  # Activation of previous layer

        # Calculate the weight derivative (including L2 regularization term)
        dW = np.matmul(dZ, A.T) / m + (lambtha / m) * weights[
            'W' + str(layerIndex)
            ]

        # Calculate the bias derivative
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # If not the first layer, calculate dZ for the next layer
        if layerIndex > 1:
            dZ = np.matmul(weights['W' + str(layerIndex)].T, dZ) * (1 - A ** 2)

        # Update weights and biases
        weights['W' + str(layerIndex)] -= alpha * dW
        weights['b' + str(layerIndex)] -= alpha * db
