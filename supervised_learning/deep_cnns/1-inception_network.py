#!/usr/bin/env python3
"""
Builds inception network as described in Going Deeper with Convolutions (2014)
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Returns: the keras model
    """
    # Initialize the input
    input_img = K.Input(shape=(224, 224, 3))

    # First Convolutional layer
    conv1 = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                            padding='same', activation='relu')(input_img)
    maxpool1 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(conv1)

    # Second Convolutional layer
    conv2 = K.layers.Conv2D(64, (1, 1), strides=(1, 1),
                            padding='same', activation='relu')(maxpool1)
    conv3 = K.layers.Conv2D(192, (3, 3), strides=(1, 1),
                            padding='same', activation='relu')(conv2)
    maxpool2 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(conv3)

    # Apply Inception blocks
    inception3a = inception_block(maxpool2, [64, 96, 128, 16, 32, 32])
    inception3b = inception_block(inception3a, [128, 128, 192, 32, 96, 64])
    maxpool3 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(inception3b)

    inception4a = inception_block(maxpool3, [192, 96, 208, 16, 48, 64])
    inception4b = inception_block(inception4a, [160, 112, 224, 24, 64, 64])
    inception4c = inception_block(inception4b, [128, 128, 256, 24, 64, 64])
    inception4d = inception_block(inception4c, [112, 144, 288, 32, 64, 64])
    inception4e = inception_block(inception4d, [256, 160, 320, 32, 128, 128])
    maxpool4 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(inception4e)

    inception5a = inception_block(maxpool4, [256, 160, 320, 32, 128, 128])
    inception5b = inception_block(inception5a, [384, 192, 384, 48, 128, 128])

    avgpool = K.layers.AveragePooling2D((7, 7), strides=(1, 1))(inception5b)
    dropout = K.layers.Dropout(0.4)(avgpool)
    output = K.layers.Dense(1000, activation='softmax')(dropout)

    model = K.models.Model(input_img, output)

    return model
