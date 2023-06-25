#!/usr/bin/env python3
"""
calculates the cost of a neural network with L2 regularization
"""


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    cost: cost of the network without L2 regularization
    lambtha: regularization parameter
    weights: dictionary of the weights and biases (numpy.ndarrays) of the neural network
    L: number of layers in the neural network
    m: number of data points used
    Returns: the cost of the network accounting for L2 regularization
    """

