#!/usr/bin/env python3
"""builds a neural network with the Keras library"""


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx is the number of input features to the network
    layers is a list containing the number of nodes in each layer of the network
    activations is a list containing the activation functions used for each layer of the network
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout
    You are not allowed to use the Sequential class
    Returns: the keras model
    """
    