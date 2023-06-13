#!/usr/bin/env python3
"""calculate accuracy function"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    calculates the accuracy of a prediction
    y: placeholder for the labels of the input data
    y_pred: tensor containing the network’s predictions
    Returns a tensor containing the decimal accuracy of the prediction
    """
    true_class_labels = tf.argmax(y, axis=1)
    predicted_class_labels = tf.argmax(y_pred, axis=1)
    return tf.reduce_mean(tf.cast(tf.math.equal(true_class_labels,
                          predicted_class_labels), tf.float32))
