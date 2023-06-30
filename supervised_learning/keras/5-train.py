#!/usr/bin/env python3
"""update to analyze validation data"""


def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False):
    """
    validation_data is the data to validate the model with, if not None
    """