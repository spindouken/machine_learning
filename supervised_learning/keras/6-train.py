#!/usr/bin/env python3
"""update the function to also train the model using early stopping"""


def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True, shuffle=False):
    """
    early_stopping is a boolean that indicates whether early stopping should be used
    early stopping should only be performed if validation_data exists
    early stopping should be based on validation loss
    patience is the patience used for early stopping
    """

