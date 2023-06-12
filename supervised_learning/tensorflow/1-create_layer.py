#!/usr/bin/env python3
"""create layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    create a layer in a neural network
    prev: tensor output of the previous layer
    n: the number of nodes in the layer to create
    activation: activation function that the layer should use
    Returns the tensor output of the layer
    """
    # He initialization for layer weights
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    layer = tf.layers.dense(inputs=prev, units=n, activation=activation,
                            kernel_initializer=initializer)
    return layer
