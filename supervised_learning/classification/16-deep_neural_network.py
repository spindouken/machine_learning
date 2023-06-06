#!/usr/bin/env python3
"""are you winning at neural networks son?"""
import numpy as np


class DeepNeuralNetwork:
    """
    defines a deep neural network performing binary classification
    """
    def __init__(self, nx, layers):
        """
        L: number of layers in the neural network
        cache: A dictionary to hold all intermediary values of the network.
            Upon instantiation, it is set to an empty dictionary
        weights: A dictionary to hold all weights and biases of the network
            Upon instantiation,
            weights initialized using He et al. method
                and saved in the weights dictionary using the key W{1}
                    where {1} is the hidden layer the weight belongs to
            biases initialized to 0's and saved
                in the weights dictionary using the key b{1}
                    where {1} is the hiddne layer the bias belongs to
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) < 0:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            if i == 0:
                self.weights['W' + str(i+1)] = np.random.randn(layers[i], nx) \
                    * np.sqrt(2/nx)
            else:
                self.weights['W' + str(i+1)] = np.random. \
                    randn(layers[i], layers[i-1]) * np.sqrt(2/layers[i-1])
            self.weights['b' + str(i+1)] = np.zeros((layers[i], 1))
