#!/usr/bin/env python3
"""creating nodes in a network.... networking....a neural network"""
import numpy as np


class NeuralNetwork:
    """
    defines a neural network with one hidden layer.
    performing binary classification.
    """
    def __init__(self, nx, nodes):
        """
        initialize a neural network
        nx: the number of input features
        nodes: the number of nodes found in the hidden layer
        W1: The weights vector for the hidden layer.
            initialized using a random normal distribution.
        b1: The bias for the hidden layer. Initialized with 0’s.
        A1: The activated output for the hidden layer. Initialized to 0.
        W2: The weights vector for the output neuron.
            Initialized using a random normal distribution.
        b2: The bias for the output neuron. Initialized to 0.
        A2: The activated output for the output neuron (prediction).
            Initialized to 0.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros(shape=(nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2
