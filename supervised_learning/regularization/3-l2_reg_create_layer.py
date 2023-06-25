#!/usr/bin/env python3
"""
creates a tensorflow layer that includes L2 regularization
"""


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    creates a tensorflow layer that includes L2 regularization
    prev: tensor containing the output of the previous layer
    n: number of nodes the new layer should contain
    activation: the activation function that should be used on the layer
    lambtha: L2 regularization parameter
    Returns: output of the new layer
    """

