#!/usr/bin/env python3
"""function that makes a prediction using a neural network"""


def predict(network, data, verbose=False):
    """
    network is the network model to make the prediction with
    data is the input data to make the prediction with
    verbose is a boolean that determines if output should be printed during the prediction process
    Returns: the prediction for the data
    """
    return network.predict(x=data,
                           verbose=verbose)
