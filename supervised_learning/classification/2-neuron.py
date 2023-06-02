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
