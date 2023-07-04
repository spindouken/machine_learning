#!/usr/bin/env python3
"""functions to save and load model's config in JSON format"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    saves a model’s configuration in JSON format
    network is the model whose configuration should be saved
    filename is the path of the file that the configuration
        should be saved to
    Returns: None
    """
    json_string = network.to_json(filename)
    with open(filename, 'w') as json:
        json.write(json_string)


def load_config(filename):
    """
    loads a model with a specific configuration
    filename is the path of the file containing the model’s
        configuration in JSON format
    Returns: the loaded model
    """
    with open(filename, 'r') as json:
        json_string = json.read()

    model = K.models.model_from_json(json_string)
    return
