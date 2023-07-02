#!/usr/bin/env python3
"""save and load model functions in keras"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    network is the model to save
    filename is the path of the file that the model should be saved to
    save function saves:
        architecture of the model, allowing it to re-recreate the model
        the weights of the model
        the training configuration (loss, optimizer)
        the state of the optimizer, allowing to resume training
            from exactly where you left off
    Returns: None
    """
    network.save(filename)


def load_model(filename):
    """
    filename is the path of the file that the model should be loaded from
    Returns: the loaded model
    """
    return K.models.load_model(filename)
