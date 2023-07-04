#!/usr/bin/env python3
"""functions to save model weights and load model weights"""


def save_weights(network, filename, save_format='h5'):
    """
    saves a model's weights
    network is the model whose weights should be saved
    filename is the path of the file that the weights should be saved to
    save_format is the format in which the weights should be saved
    Returns: None
    """
    network.save_weights(filepath=filename, save_format=save_format)


def load_weights(network, filename):
    """
    loads a model's weights
    network is the model whose weights should be loaded
    filename is the path of the file that the weights
        should be loaded from
    returns none
    """
    network.load_weights(filepath=filename)
