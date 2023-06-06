#!/usr/bin/env python3
"""creating nodes in a network.... networking....a neural network"""
import numpy as np


class NeuralNetwork:
    """
    defines a neural network with one hidden layer
    performing binary classification
    """
    def __init__(self, nx, nodes):
        """
        initialize a neural network
        nx: the number of input features
        nodes: the number of nodes found in the hidden layer
        W1: The weights vector of the hidden layer.
            initialized using a random normal distribution.
        b1: The bias of the hidden layer. Initialized with 0’s.
        A1: The activated output of the hidden layer. Initialized to 0.
        W2: The weights vector of the output neuron.
            Initialized using a random normal distribution.
        b2: The bias of the output neuron. Initialized to 0.
        A2: The activated output of the output neuron (prediction).
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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        X: numpy.ndarray of shape (nx, m) that contains the input data
        nx: number of input features to the neuron
        m: number of examples
        Returns the private attributes __A1 and __A2
        """
        # Z1 is the result of the dot product
        #   of weights and input data plus the bias
        # represents the input of the activation function of the hidden layer
        Z1 = np.matmul(self.__W1, X) + self.__b1

        # Apply the sigmoid activation function to Z1 to get A1
        # The sigmoid function transforms the input to a value
        #   between 0 and 1 which can be interpreted as a probability
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # Z2 is the result of the dot product
        #   of weights and the output of the hidden layer plus the bias
        # represents the input of the activation function of the output neuron
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2

        # Apply the sigmoid activation function to Z2 to get A2
        # The sigmoid function transforms the input to a value
        #   between 0 and 1 which can be interpreted as a probability
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Y: numpy.ndarray with shape (1, m)
            that contains the correct labels of the input data
        A: numpy.ndarray with shape (1, m)
            containing the activated output of the neuron of each example
        Returns the cost of the model
        """
        # Number of examples
        m = Y.shape[1]

        # Calculation of the cost using the logistic regression cost function
        # Note: To avoid division by zero errors,
        #   we use 1.0000001 - A instead of 1 - A
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        Parameters:
        X: numpy.ndarray with shape (nx, m) that contains the input data
        nx: number of input features to the neuron
        m: number of examples
        Y: numpy.ndarray with shape (1, m)
            that contains the correct labels of the input data
        Returns the neuron’s prediction and the cost of the network
        """

        # Forward propagation to get the activated output
        A1, A2 = self.forward_prop(X)

        # Calculate the cost of the network
        cost = self.cost(Y, A2)

        # The prediction should be a numpy.ndarray with shape (1, m)
        # containing the predicted labels of each example
        # The label values should be 1
        #    if the output of the network is >= 0.5 and 0 otherwise
        prediction = np.where(A2 >= 0.5, 1, 0)

        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        X: numpy.ndarray with shape (nx, m) that contains the input data
        Y: numpy.ndarray with shape (1, m)
            that contains the correct labels of the input data
        A1: output of the hidden layer
        A2: predicted output
        alpha: learning rate
        Updates the private attributes __W1, __b1, __W2, and __b2
        """
        m = X.shape[1]
        # Calculate the dif between the predicted output and the actual output
        dZ2 = A2 - Y
        # Calculate the derivative of the cost with respect to W2
        dW2 = np.dot(dZ2, A1.T) / m
        # derivative of the cost with respect to b2
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        # derivative of the cost with respect to Z1
        dZ1 = np.dot(self.__W2.T, dZ2) * (A1 * (1 - A1))
        # derivative of the cost with respect to W1
        dW1 = np.dot(dZ1, X.T) / m
        # derivative of the cost with respect to b1
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Update W1 by subtracting the product of the learning rate and dW1
        self.__W1 = self.__W1 - alpha * dW1
        # Update b1
        self.__b1 = self.__b1 - alpha * db1
        # Update W2
        self.__W2 = self.__W2 - alpha * dW2
        # Update b2
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network
        X: numpy.ndarray with shape (nx, m) that contains the input data
        Y: numpy.ndarray with shape (1, m)
            that contains the correct labels for the input data
        iterations: number of iterations to train over
        alpha: learning rate
        Returns the evaluation of the training data after iterations complete
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # Perform the training iterations
        for _ in range(iterations):
            # Forward propagation
            A1, A2 = self.forward_prop(X)
            # Cost calculation
            cost = self.cost(Y, A2)
            # Backpropagation (gradient descent)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)
