#!/usr/bin/env python3
"""
creates an autoencoder
"""
import tensorflow.keras as keras


def buildEncoder(input_dims, hidden_layers, latent_dims):
    """build encoder"""
    encoderInput = keras.Input(shape=(input_dims,))
    x = encoderInput

    for units in hidden_layers:
        x = keras.layers.Dense(units=units, activation="relu")(x)

    encoderOutput = keras.layers.Dense(
        units=latent_dims, activation="relu"
    )(x)

    return (
        keras.Model(inputs=encoderInput, outputs=encoderOutput),
        encoderInput,
    )


def buildDecoder(latent_dims, hidden_layers, output_dims):
    """build decoder"""
    decoderInput = keras.Input(shape=(latent_dims,))
    x = decoderInput

    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units=units, activation="relu")(x)

    decoderOutput = keras.layers.Dense(
        units=output_dims, activation="sigmoid"
    )(x)

    return keras.Model(inputs=decoderInput, outputs=decoderOutput)


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    input_dims is an integer containing
        the dimensions of the model input
    hidden_layers is a list containing
        the number of nodes for each hidden layer
        in the encoder, respectively
    the hidden layers should be reversed for the decoder
    latent_dims is an integer containing
        the dimensions of the latent space representation
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled
        using adam optimization and binary cross-entropy loss
    All layers should use a relu activation
        except for the last layer in the decoder,
        which should use sigmoid
    """
    encoder, encoderInput = buildEncoder(
        input_dims, hidden_layers, latent_dims
    )
    decoder = buildDecoder(latent_dims, hidden_layers, input_dims)

    encodedOutput = encoder(encoderInput)
    decodedOutput = decoder(encodedOutput)

    auto = keras.Model(inputs=encoderInput, outputs=decodedOutput)
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
