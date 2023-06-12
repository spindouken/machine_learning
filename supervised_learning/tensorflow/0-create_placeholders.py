#!/usr/bin/env python3
"""function that returns two placeholders, x and y, for the network"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """create placeholders x and y"""
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return x, y
