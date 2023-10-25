#!/usr/bin/env python3
"""
creates a convolutional autoencoder
"""
import tensorflow.keras as keras


def buildEncoder(input_dims, filters):
    """build encoder"""
    encoderInput = keras.Input(shape=input_dims)
    x = encoderInput

    for f in filters:
        x = keras.layers.Conv2D(
            f, (3, 3), activation="relu", padding="same"
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)

    encoderOutput = x

    return (
        keras.Model(inputs=encoderInput, outputs=encoderOutput),
        encoderInput,
    )


def buildDecoder(latent_dims, filters):
    """build decoder"""
    decoderInput = keras.Input(shape=latent_dims)
    x = decoderInput

    x = keras.layers.Conv2D(
        filters[2], (3, 3), activation="relu", padding="same"
    )(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(
        filters[1], (3, 3), activation="relu", padding="same"
    )(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(
        filters[0], (3, 3), activation="relu", padding="valid"
    )(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    decoderOutput = keras.layers.Conv2D(
        1, (3, 3), activation="sigmoid", padding="same"
    )(x)

    return keras.Model(inputs=decoderInput, outputs=decoderOutput)


def autoencoder(input_dims, filters, latent_dims):
    """
    input_dims is a tuple of integers containing the dimensions of the model
        input
    filters is a list containing the number of filters for each convolutional
        layer in the encoder, respectively
            the filters should be reversed for the decoder
    latent_dims is a tuple of integers containing the dimensions of the latent
        space representation
    Each convolution in the encoder should use a kernel size of (3, 3) with
        same padding and relu activation, followed by max pooling of size (2,
        2)
    Each convolution in the decoder, except for the last two, should use a
        filter size of (3, 3) with same padding and relu activation, followed
        by upsampling of size (2, 2)
        The second to last convolution should instead use valid padding
        The last convolution should have the same number of filters as the
            number of channels in input_dims with sigmoid activation and no
            upsampling
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and
        binary cross-entropy loss
    """
    encoder, encoderInput = buildEncoder(input_dims, filters)
    decoder = buildDecoder(latent_dims, filters)

    encodedOutput = encoder(encoderInput)
    decodedOutput = decoder(encodedOutput)

    auto = keras.Model(inputs=encoderInput, outputs=decodedOutput)
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
