#!/usr/bin/env python3
"""
creates a tensorflow layer that includes L2 regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    creates a tensorflow layer that includes L2 regularization
    prev: tensor containing the output of the previous layer
    n: number of nodes the new layer should contain
    activation: the activation function that should be used on the layer
    lambtha: L2 regularization parameter
    Returns the output of the new layer
    """
    # Create an L2 regularizer with the given regularization parameter
    regularization = tf.contrib.layers.l2_regularizer(lambtha)

    # Create a variance scaling initializer
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    # Create a dense layer with the specified number of nodes, activation function,
    # kernel regularizer, and kernel initializer
    l2_regularized_layer = tf.layers.Dense(
        n, activation=activation,
        kernel_regularizer=regularization, kernel_initializer=initialize
        )

    # Connect the new layer to the previous layer and return the output
    return l2_regularized_layer(prev)
