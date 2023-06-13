#!/usr/bin/env python3
"""forward propagation function"""
import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    creates the forward propagation graph for the neural network
    x is the placeholder for the input data
    layer_sizes: list containing the number of nodes
        in each layer of the network
    activations: list containing the activation functions
        for each layer of the network
    Returns the prediction of the network in tensor form
    """
    create_layer = __import__('1-create_layer').create_layer
    output = x
    for i in range(len(layer_sizes)):
        output = create_layer(output, layer_sizes[i], activations[i])
        prediction = output
    return prediction
