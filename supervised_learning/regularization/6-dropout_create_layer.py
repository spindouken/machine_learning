#!/usr/bin/env python3
"""
Function that creates a layer of a neural network using dropout
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    prev: a tensor containing the output of the previous layer
    n: number of nodes the new layer should contain
    activation: the activation function that should be used on the layer
    keep_prob: the probability that a node will be kept
    Returns the output of the new layer
    """
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dropout = tf.layers.Dropout(rate=keep_prob)
    layer = tf.layers.dense(inputs=prev,
                            units=n,
                            activation=activation,
                            kernel_initializer=initialize,
                            kernel_regularizer=dropout)
    return layer
