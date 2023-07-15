#!/usr/bin/env python3
"""
builds an inception block as described in Going Deeper with Convolutions (2014)
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP:
        F1 is the number of filters in the 1x1 convolution
        F3R is the number of filters in 1x1 convolution before 3x3 convolution
        F3 is the number of filters in 3x3 convolution
        F5R is the number of filters in 1x1 convolution before 5x5 convolution
        F5 is the number of filters in the 5x5 convolution
        FPP is the number of filters in 1x1 convolution after the max pooling
    All conv's inside inception block use rectified linear activation (ReLU)
    Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 Convolution block
    conv_1x1 = K.layers.Conv2D(F1, 1, activation='relu')(A_prev)

    # 3x3 Convolution block: 1x1 conv reduces filter number,
    # 3x3 conv applies filters
    conv_3x3 = K.layers.Conv2D(F3R, 1, activation='relu')(A_prev)
    conv_3x3 = K.layers.Conv2D(F3, 3, padding='same',
                               activation='relu')(conv_3x3)

    # 5x5 Convolution block: 1x1 conv reduces filter number,
    # 5x5 conv applies filters
    conv_5x5 = K.layers.Conv2D(F5R, 1, activation='relu')(A_prev)
    conv_5x5 = K.layers.Conv2D(F5, 5, padding='same',
                               activation='relu')(conv_5x5)

    # Pool Projection block: 3x3 max pooling,
    # 1x1 conv changes filter number
    pool_proj = K.layers.MaxPooling2D(3, 1, padding='same')(A_prev)
    pool_proj = K.layers.Conv2D(FPP, 1, activation='relu')(pool_proj)

    # Concatenation of all filters
    output = K.layers.concatenate([conv_1x1,
                                  conv_3x3,
                                  conv_5x5,
                                  pool_proj],
                                  axis=3)

    return output
