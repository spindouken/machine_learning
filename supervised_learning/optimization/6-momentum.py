#!/usr/bin/env python3
"""function that creates the training operation for a neural network in
tensorflow using the gradient descent with momentum optimization algorithm"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    loss: loss of the network
    alpha: learning rate
    beta1: momentum weight
    Returns the momentum optimization operation
    """
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    train_op = optimizer.minimize(loss)
