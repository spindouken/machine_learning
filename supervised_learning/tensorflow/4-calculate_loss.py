#!/usr/bin/env python3
"""calculate loss"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    calculates the softmax cross-entropy loss of a prediction
    y: placeholder for the labels of the input data
    y_pred: a tensor containing the network’s predictions
    Returns a tensor containing the loss of the prediction
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
