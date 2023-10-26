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


class VAELossLayer(keras.layers.Layer):
    """VAE loss layer"""

    @staticmethod
    def VAELoss(x, xDecodedMean, zLogSigma, zMean):
        """calculate VAE loss"""
        xLoss = keras.backend.binary_crossentropy(x, xDecodedMean)
        xLoss = keras.backend.sum(xLoss, axis=1)
        klLoss = -0.5 * keras.backend.mean(
            1
            + zLogSigma
            - keras.backend.square(zMean)
            - keras.backend.exp(zLogSigma),
            axis=-1,
        )
        return keras.backend.mean(xLoss + klLoss)

    def call(self, inputs):
        """call method to calculate VAE loss and add it to the layer"""
        x, xDecodedMean, zLogSigma, zMean = inputs
        loss = self.VAELoss(x, xDecodedMean, zLogSigma, zMean)
        self.add_loss(loss, inputs=inputs)
        return x


def sampling(args):
    """perform sampling"""
    zMean, zLogSigma = args
    epsilon = keras.backend.random_normal(shape=keras.backend.shape(zMean))
    return zMean + keras.backend.exp(zLogSigma / 2) * epsilon


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
    encoderInput, zMean, zLogSigma = buildEncoder(
        input_dims, hidden_layers, latent_dims
    )
    decoderInput, decoderOutput = buildDecoder(
        latent_dims, hidden_layers, input_dims
    )

    encoder = keras.Model(inputs=encoderInput, outputs=[zMean, zLogSigma])
    decoder = keras.Model(inputs=decoderInput, outputs=decoderOutput)

    autoInput = keras.Input(shape=(input_dims,))
    zMeanAuto, zLogSigmaAuto = encoder(autoInput)

    z = keras.layers.Lambda(sampling)([zMeanAuto, zLogSigmaAuto])

    autoOutput = decoder(z)
    vaeLossLayer = VAELossLayer()(
        [autoInput, autoOutput, zLogSigmaAuto, zMeanAuto]
    )

    auto = keras.Model(inputs=autoInput, outputs=vaeLossLayer)
    auto.compile(optimizer="adam")

    return encoder, decoder, auto
