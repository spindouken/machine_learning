#!/usr/bin/env python3
"""
Builds a dense block as described in Densely Connected Convolutional Networks
This architecture is unique because it allows
    feature maps to flow from any layer to any subsequent layer
This creates more direct paths for gradient during
    backpropagation and strengthens feature propagation
"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    The Dense Block is a set of layers,
        each connected to all others in a feed-forward way.
    For each layer, feature maps of all preceding layers are used as inputs,
    and its own feature maps are used as inputs into all subsequent layers.

    Args:
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block

    Returns:
    The concatenated output of each layer within the Dense Block
        and the number of filters within the concatenated outputs
    """
    # Initialize weights using He Normal initialization
    HeNormal = K.initializers.he_normal()

    # Iterate for the number of layers
    for i in range(layers):

        # First, we perform batch normalization on the channels axis,
        #   then apply ReLU activation
        Z = K.layers.BatchNormalization(axis=3)(X)
        Z = K.layers.Activation('relu')(Z)

        # Bottleneck convolution: 1x1 convolution,
        #   reduces dimensionality, here we use 4 * growth rate filters
        Z = K.layers.Conv2D(filters=4 * growth_rate, kernel_size=1,
                            padding='same',
                            kernel_initializer=HeNormal)(Z)

        # Again, batch normalization followed by ReLU activation
        Z = K.layers.BatchNormalization(axis=3)(Z)
        Z = K.layers.Activation('relu')(Z)

        # Second Convolution, a 3x3 convolution that
        #   will produce growth_rate feature maps
        Z = K.layers.Conv2D(filters=growth_rate, kernel_size=3, padding='same',
                            kernel_initializer=HeNormal)(Z)

        # Concatenate the input (previous layer) and the bottleneck layer,
        #   this is the 'dense' connection
        X = K.layers.concatenate([X, Z])

        # Increase the number of filters for the next layer by growth_rate
        nb_filters += growth_rate

    # Returns the concatenated output of each layer within the
    #   dense block and the number of filters within the concatenated outputs
    return X, nb_filters
