#!/usr/bin/env python3
"""
creates a batch normalization layer for a neural network in tensorflow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    prev: activated output of the previous layer
    n: number of nodes in the layer to be created
    activation: activation function that should be used
        on the output of the layer
    Returns: a tensor of the activated output for the layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=initializer)
    Z = layer(prev)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name="gamma")
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name="beta")
    mean, variance = tf.nn.moments(Z, axes=0)
    epsilon = 1e-8
    Znorm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, epsilon)
    return activation(Znorm)
