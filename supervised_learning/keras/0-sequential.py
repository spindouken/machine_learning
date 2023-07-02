#!/usr/bin/env python3
"""builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx is the number of input features to the network
    layers is a list containing the number
        of nodes in each layer of the network
    activations is a list containing the activation functions
    used for each layer of the network
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout
    Returns: the keras model
    """
    # initialize keras sequential model
    model = K.models.Sequential()

    for i in range(len(layers)):

        if i == 0:
            model.add(
                K.layers.Dense(layers[i], input_dim=nx,
                               activation=activations[i],
                               kernel_regularizer=K.regularizers.l2(lambtha))
                      )
        else:
            model.add(
                K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=K.regularizers.l2(lambtha))
                      )
        # add dropout after each layer except the last one
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
