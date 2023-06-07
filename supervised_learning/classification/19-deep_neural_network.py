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
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: x > 0 and isinstance(x, int), layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__nx = nx
        self.__layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            if i == 0:
                self.__weights['W' + str(i+1)] = np.random. \
                    randn(layers[i], nx) * np.sqrt(2/nx)
            else:
                self.__weights['W' + str(i+1)] = np.random. \
                    randn(layers[i], layers[i-1]) * np.sqrt(2/layers[i-1])
            self.__weights['b' + str(i+1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        # Save the input data to the cache
        self.__cache['A0'] = X

        # Loop over each layer in the network
        for i in range(self.__L):
            # Retrieve the weights and bias of the current layer
            W = self.__weights['W' + str(i+1)]
            b = self.__weights['b' + str(i+1)]

            # Retrieve the activations from the previous layer
            A_prev = self.__cache['A' + str(i)]

            # Calculate the weighted sum of inputs and bias of current layer
            Z = np.matmul(W, A_prev) + b

            # Apply the sigmoid activation function
            # The sigmoid function maps any input to a value between 0 and 1
            A = 1 / (1 + np.exp(-Z))

            # Save the output of the current layer to the cache
            self.__cache['A' + str(i+1)] = A

        # Return the output of the last layer and the cache
        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Y: a numpy.ndarray with shape (1, m)
            that contains the correct labels of the input data
        A: a numpy.ndarray with shape (1, m)
            containing the activated output of the neuron for each example
        Returns the cost of the model as a float
        """
        # m is the number of examples
        m = Y.shape[1]

        # Calculate the loss for each training example
        #    using the formula for cross-entropy loss
        # The term 1.0000001 - A is used instead of 1 - A
        #    to avoid division by zero errors when A is exactly 1
        loss = - (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        # Calculate the cost as the average of the losses
        cost = np.sum(loss) / m

        return cost
