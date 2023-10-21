#!/usr/bin/env python3
"""
creates a variational autoencoder
"""
import tensorflow.keras as K
import tensorflow as tf


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
