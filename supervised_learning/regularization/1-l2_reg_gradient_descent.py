#!/usr/bin/env python3
"""
updates the weight and biases of a neural network
using gradient descent with L2 regularization
"""


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

