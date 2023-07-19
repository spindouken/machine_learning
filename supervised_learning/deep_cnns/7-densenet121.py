#!/usr/bin/env python3
"""
Builds the DenseNet-121 architecture as described
    in Densely Connected Convolutional Networks
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    builds DenseNet-121 architecture

    growth_rate is the growth rate
    compression is the compression factor

    input data will have shape (224, 224, 3)
    All convolutions preceded by Batch Normalization ReLU
    All weights use he normal initialization

    Returns: the keras model
    """
    HeNormal = K.initializers.he_normal()
    # Define input layer
    inputs = K.Input(shape=(224, 224, 3))

    # Initial convolution
    nb_filters = 2 * growth_rate
    X = K.layers.BatchNormalization(axis=3)(inputs)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(filters=nb_filters, kernel_size=7, strides=2,
                        padding='same', kernel_initializer=HeNormal)(X)

    # Pooling layer
    X = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(X)

    # Dense block
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 6)

    # Transition layer 1
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense block 2
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)

    # Transition layer 2
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense block 3
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)

    # Transition layer 3
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense block 4
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Global Average Pooling
    X = K.layers.AveragePooling2D(pool_size=7, strides=1, padding='valid')(X)
    X = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer=HeNormal)(X)

    model = K.models.Model(inputs=inputs, outputs=X)

    return model
