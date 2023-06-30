#!/usr/bin/env python3
"""
update train model to also save the best iteration of the model
"""


def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    save_best is a boolean indicating whether to save the model after each epoch if it is the best
    a model is considered the best if its validation loss is the lowest that the model has obtained
    filepath is the file path where the model should be saved
    """

