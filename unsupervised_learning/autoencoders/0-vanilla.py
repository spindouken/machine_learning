#!/usr/bin/env python3
"""
creates an autoencoder
"""


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates an autoencoder
    :param input_dims: is an integer containing the dimensions of the model
    input
    :param hidden_layers: is a list containing the number of nodes for each
    hidden layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    :param latent_dims: is an integer containing the dimensions of the latent
    space representation
    :return: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    """
