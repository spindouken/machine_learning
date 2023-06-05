#!/usr/bin/env python3
"""look pa, I'm makin neurons"""
import numpy as np


class Neuron:
    """defines a single neuron performing binary classification"""
    def __init__(self, nx):
        """
        W: The weights vector for the neuron
        b: The bias for the neuron
        A: The activated output of the neuron (prediction)
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter for W"""
        return self.__W

    @property
    def b(self):
        """getter for b"""
        return self.__b

    @property
    def A(self):
        """getter for A"""
        return self.__A

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neuron
        X is a numpy.darray with a shape (nx, m) contains the input data
        calculate weighted sum of inputs + bias (Z = WX + b)
        utilize sigmoid activation function (sigmoid(Z) = 1 / (1 + e^-Z))
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Y: contains the correct labels for the input data...
        ...and is a numpy.ndarray with shape (1, m)
        A: contains the activated output of the neuron for each example...
        ...and is a numpy.ndarray with shape (1, m)
        m: the number of examples
        Σ: the sum over all examples
        To avoid division by zero errors, use 1.0000001 - A instead of 1 - A
        Returns the cost

        Cost function for logistic regression:
        cost = -1/m * Σ [Y * log(A) + (1 - Y) * log(1 - A)]
        """
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        X: numpy.ndarray with shape (nx, m) that contains the input data
        Y: numpy.ndarray with shape (1, m)
            that contains the correct labels for the input data
        Returns the neuron’s prediction and the cost of the network
        """
        # Generate predictions
        A = self.forward_prop(X)
        # Calculate cost
        cost = self.cost(Y, A)
        # Apply threshold to generate binary predictions
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
