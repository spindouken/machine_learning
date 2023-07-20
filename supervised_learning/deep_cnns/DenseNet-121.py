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
    # He Normal initialization will be used to initialize weights
    HeNormal = K.initializers.he_normal()
    # Define input
    inputs = K.Input(shape=(224, 224, 3))

    # Initial convolution
    nb_filters = 2 * growth_rate
    X = K.layers.BatchNormalization(axis=3)(inputs)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(filters=nb_filters, kernel_size=7, strides=2,
                        padding='same', kernel_initializer=HeNormal)(X)
    X = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(X)

    # Dense block
    # output from first initial setup (convolution, pooling, etc) enters
    #   the first dense block
    # Batch normalization -> ReLU -> 1x1 Convolution (bottleneck layer)
    #   -> Another batch normalization -> ReLU -> 3x3 Convolution
    # The output feature-maps of the 3x3 convolution are concatenated with
    #   the previous feature-maps. This is repeated 6 times in the first block.
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 6)

    # Transition layer 1
    # After the dense block, the feature maps go through a transition layer,
    #   which reduces number of feature-maps & downsamples spatial dimensions
    # Batch normalization -> ReLU -> 1x1 Convolution -> 2x2 Average Pooling
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense block 2
    # similar to first dense block, but repeated 12 times
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)

    # Transition layer 2
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense block 3
    # repeated 24 times
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)

    # Transition layer 3
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense block 4
    # repeated 16 times
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Global Average Pooling
    # reduces spatial dimensions to 1x1
    # followed by fully connected (dense) layer with 1000 units
    #   (for 1000-class classification) and a softmax activation function
    #   to output the probabilities for each class  
    X = K.layers.AveragePooling2D(pool_size=7, strides=1, padding='valid')(X)
    X = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer=HeNormal)(X)

    model = K.models.Model(inputs=inputs, outputs=X)

    return model

