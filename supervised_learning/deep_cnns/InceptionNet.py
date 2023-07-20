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


def inception_network():
    """
    Builds inception network as described in Going Deeper with Convolutions (2014)
    Returns: the keras model
    """
    input_layer = K.Input(shape=(224, 224, 3))

    # initial layers to reduce size of input and extract low-level features
    conv_7x7 = K.layers.Conv2D(filters=64,
                               padding='same',
                               activation='relu',
                               kernel_size=(7, 7),
                               strides=(2, 2))(input_layer)
    maxpool_3x3_1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                          strides=(2, 2),
                                          padding='same')(conv_7x7)
    conv_3x3 = K.layers.Conv2D(filters=192,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='same',
                               activation='relu')(maxpool_3x3_1)
    maxpool_3x3_2 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                          strides=(2, 2),
                                          padding='same')(conv_3x3)

    # inception blocks
    # each inception block takes the output of the previous layer as input and
    #   produces a concatenated output of several convolutional layers and
    #   max pooling layers, the number of filters in each convolutional layer
    #   is specified in the filters parameter (A_prev, filters)
    inception_3a = inception_block(maxpool_3x3_2, (64, 96, 128, 16, 32, 32))
    inception_3b = inception_block(inception_3a, (128, 128, 192, 32, 96, 64))
    maxpool_3x3_3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                          strides=(2, 2),
                                          padding='same')(inception_3b)

    inception_4a = inception_block(maxpool_3x3_3, (192, 96, 208, 16, 48, 64))
    inception_4b = inception_block(inception_4a, (160, 112, 224, 24, 64, 64))
    inception_4c = inception_block(inception_4b, (128, 128, 256, 24, 64, 64))
    inception_4d = inception_block(inception_4c, (112, 144, 288, 32, 64, 64))
    inception_4e = inception_block(inception_4d, (256, 160, 320, 32, 128, 128))
    maxpool_3x3_4 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                          strides=(2, 2),
                                          padding='same')(inception_4e)

    inception_5a = inception_block(
        maxpool_3x3_4, (256, 160, 320, 32, 128, 128))
    inception_5b = inception_block(inception_5a, (384, 192, 384, 48, 128, 128))
    # avg pool to help reduce the size of the feature map
    avg_pool_7x7 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                             strides=(1, 1),
                                             padding='valid')(inception_5b)
    # dropout to help reduce overfitting
    dropout_40 = K.layers.Dropout(rate=(0.4))(avg_pool_7x7)
    # fully connected layer can produce the final output of the network
    output = K.layers.Dense(units=(1000), activation='softmax')(dropout_40)

    model = K.Model(inputs=input_layer, outputs=output)
    # Inception network is trained using Adam optimzer
    #   and categorical crossentropy loss
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
