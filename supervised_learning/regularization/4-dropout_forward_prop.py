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
    keep_prob: the probability that A node will be kept
    Returns A dictionary containing the outputs of each layer and the dropout
        mask used on each layer
    """
    outputs = {}
    outputs["A0"] = X
    for layer in range(1, L+1):
        W = weights["W{}".format(layer)]
        A = outputs["A{}".format(layer-1)]
        b = weights["b{}".format(layer)]
        Z = np.matmul(W, A) + b

        if layer == L:
            exponentiated_values = np.exp(Z)
            outputs["A{}".format(layer)] = exponentiated_values / np.sum(
                exponentiated_values, axis=0
                )

        else:
            top = np.exp(Z) - np.exp(-Z)
            bot = np.exp(Z) + np.exp(-Z)
            A = top / bot

            dX = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            outputs["D{}".format(layer)] = dX*1
            A *= dX
            A /= keep_prob
            outputs["A{}".format(layer)] = A

    return outputs
