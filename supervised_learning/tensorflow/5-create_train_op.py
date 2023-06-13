#!/usr/bin/env python3
"""Create train operation"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    creates the training operation for the network
    loss: loss of the network’s prediction
    alpha: the learning rate
    Returns an operation that trains the network using gradient descent
    """
    gdoptimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = gdoptimizer.minimize(loss)
    return train_op
