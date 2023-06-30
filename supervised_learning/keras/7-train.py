#!/usr/bin/env python3
"""
update the function to also train the model with learning rate decay
"""


def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    """
    learning_rate_decay is a boolean that indicates whether learning rate decay should be used
    learning rate decay should only be performed if validation_data exists
    the decay should be performed using inverse time decay
    the learning rate should decay in a stepwise fashion after each epoch
    each time the learning rate updates, Keras should print a message
    alpha is the initial learning rate
    decay_rate is the decay rate
    """
    
