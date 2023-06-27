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
    regularization = tf.contrib.layers.l2_regularizer(regularization_parameter)
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2_regularized_layer = tf.layers.Dense(number_of_nodes,
                                           activation=activation_function,
                                           kernel_regularizer=regularization,
                                           kernel_initializer=initializer)
    return l2_regularized_layer(previous_layer)
