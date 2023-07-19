#!/usr/bin/env python3
"""
Builds a transition layer as described
    in Densely Connected Convolutional Networks
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    builds a transition layer

    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer

    Code implements compression as used in DenseNet-C
    All weights use he normal initialization
    All convolutions preceded by Batch Normalization and ReLU

    Returns: The output of the transition layer
        and the number of filters within the output
    """
    HeNormal = K.initializers.he_normal()

    # Batch Normalization -> ReLU -> 1x1 Convolution -> Average Pooling
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Implement compression
    nb_filters = int(nb_filters * compression)
    X = K.layers.Conv2D(filters=nb_filters, kernel_size=1, strides=1,
                        padding='same', kernel_initializer=HeNormal)(X)

    # Average pooling with stride 2 (this will reduce
    #   the spatial dimensions of the output volume by half)
    X = K.layers.AveragePooling2D(pool_size=2, strides=2, padding='valid')(X)

    return X, nb_filters
