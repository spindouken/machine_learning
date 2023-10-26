#!/usr/bin/env python3
"""
creates a variational autoencoder
"""
import tensorflow.keras as keras
import tensorflow as tf


def buildLayers(inputTensor, layerUnits, activation="relu"):
    """build layers"""
    for units in layerUnits:
        inputTensor = keras.layers.Dense(
            units=units, activation=activation
        )(inputTensor)
    return inputTensor


def buildEncoder(input_dims, hidden_layers, latent_dims):
    """build encoder"""
    encoderInput = keras.Input(shape=(input_dims,))
    encoderHidden = buildLayers(encoderInput, hidden_layers)
    zMean = keras.layers.Dense(units=latent_dims, activation=None)(
        encoderHidden
    )
    zLogSigma = keras.layers.Dense(units=latent_dims, activation=None)(
        encoderHidden
    )
    return encoderInput, zMean, zLogSigma


def buildDecoder(latent_dims, hidden_layers, outputDims):
    """build decoder"""
    decoderInput = keras.Input(shape=(latent_dims,))
    decoderHidden = buildLayers(decoderInput, reversed(hidden_layers))
    decoderOutput = keras.layers.Dense(
        units=outputDims, activation="sigmoid"
    )(decoderHidden)
    return decoderInput, decoderOutput


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
        layer in the encoder, respectively
            the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
        representation
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and
        binary cross-entropy loss
    All layers should use a relu activation except for the last layer in the
        decoder, which should use sigmoid
    """
