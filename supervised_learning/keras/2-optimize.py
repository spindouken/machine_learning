#!/usr/bin/env python3
"""ets up Adam optimization for a keras model with categorical crossentropy loss and accuracy metrics"""


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    network is the model to optimize
    alpha is the learning rate
    beta1 is the first Adam optimization parameter
    beta2 is the second Adam optimization parameter
    Returns: None
    """

