#!/usr/bin/env python3
"""
Builds identity block as described in
    Deep Residual Learning for Image Recognition (2015)
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
    Returns: the activated output of the identity block
    """
    # Unpack the number of filters for each layer
    F11, F3, F12 = filters

    # Save the input value... will need later to add back to main path
    X_shortcut = A_prev

    # First component of main path (1x1 convolution, down-sampling)
    X = K.layers.Conv2D(filters=F11, kernel_size=(
        1, 1), strides=(1, 1), padding='valid')(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X_downsampled = K.layers.Activation('relu')(X)

    # Second component of main path (3x3 convolution, processing)
    X = K.layers.Conv2D(filters=F3, kernel_size=(
        3, 3), strides=(1, 1), padding='same')(X_downsampled)
    X = K.layers.BatchNormalization(axis=3)(X)
    X_processed = K.layers.Activation('relu')(X)

    # Third component of main path (1x1 convolution, up-sampling)
    X = K.layers.Conv2D(filters=F12, kernel_size=(
        1, 1), strides=(1, 1), padding='valid')(X_processed)
    X = K.layers.BatchNormalization(axis=3)(X)
    X_upsampled = X

    # Final step: Add shortcut value to main path, then pass through RELU act
    X_final = K.layers.Add()([X_upsampled, X_shortcut])
    X_activated = K.layers.Activation('relu')(X_final)

    return X_activated
