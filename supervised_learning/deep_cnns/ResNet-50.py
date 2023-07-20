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

    # Define the He Normal (VarianceScaling) initializer
    HeNormal = K.initializers.VarianceScaling(
        scale=2.0, mode='fan_in', distribution='truncated_normal')

    # First component of main path (1x1 convolution, down-sampling)
    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', kernel_initializer=HeNormal)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X_downsampled = K.layers.Activation('relu')(X)

    # Second component of main path (3x3 convolution, processing)
    X = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), strides=(
        1, 1), padding='same', kernel_initializer=HeNormal)(X_downsampled)
    X = K.layers.BatchNormalization(axis=3)(X)
    X_processed = K.layers.Activation('relu')(X)

    # Third component of main path (1x1 convolution, up-sampling)
    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', kernel_initializer=HeNormal)(X_processed)
    X = K.layers.BatchNormalization(axis=3)(X)
    X_upsampled = X

    # Final step: Add shortcut value to main path, then pass through RELU act
    X_final = K.layers.Add()([X_upsampled, X_shortcut])
    X_activated = K.layers.Activation('relu')(X_final)

    return X_activated

def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block

    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution,
            as well as the 1x1 convolution in the shortcut connection
    s is the stride of the first convolution
        in both the main path and the shortcut connection
    Returns: the activated output of the projection block
    """
    # Unpack the number of filters for each layer
    F11, F3, F12 = filters

    # Save the input value
    X_shortcut = A_prev

    HeNormal = K.initializers.he_normal(seed=None)

    # MAIN PATH
    # First component of main path
    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(s, s),
                        padding='valid', kernel_initializer=HeNormal)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component of main path
    X = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), strides=(
        1, 1), padding='same', kernel_initializer=HeNormal)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component of main path
    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', kernel_initializer=HeNormal)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # SHORTCUT PATH
    X_shortcut = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(
        s, s), padding='valid', kernel_initializer=HeNormal)(X_shortcut)
    X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)

    # Final step: Add shortcut value to main path,
    #   and pass it through a RELU activation
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X

def resnet50():
    """
    Builds the ResNet-50 architecture

    You can assume the input data will have shape (224, 224, 3)
    All weights use he normal initialization
    Returns: the keras model
    """
    HeNormal = K.initializers.he_normal()
    X_input = K.layers.Input((224, 224, 3))

    X = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding="same",
        kernel_initializer=HeNormal,
    )(X_input)
    # normalize before activation function
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    # pool one
    X = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same",
    )(X)
    # all of above is to reduce spatial dimensions while
    #   retaining complexity

    # The convolution blocks 2,3,4,5 with projection and identity blocks
    # Each of these blocks will output feature maps of dif dimensions (filters)
    # conv2 block
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # conv3 block
    X = projection_block(X, [128, 128, 512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # conv4 block
    X = projection_block(X, [256, 256, 1024], s=2)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # conv5 block
    X = projection_block(X, [512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # avg pool
    X = K.layers.AveragePooling2D(
        pool_size=7,
        strides=1,
        padding="valid",
    )(X)

    X = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer=HeNormal)(X)

    model = K.models.Model(inputs=X_input, outputs=X)

    return model
