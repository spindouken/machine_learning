#!/usr/bin/env python3
"""placeholder"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    placeholder
    """
    optimizer = K.optimizers.Adam(learning_rate=alpha,
                                       beta_1=beta1,
                                       beta_2=beta2)
    network.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=["accuracy"])
    return None
