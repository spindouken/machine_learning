#!/usr/bin/env python3
"""look pa, I'm makin neurons"""
import numpy as np
import matplotlib.pyplot as plt


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
        """getter of W"""
        return self.__W

    @property
    def b(self):
        """getter of b"""
        return self.__b

    @property
    def A(self):
        """getter of A"""
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
        Y: contains the correct labels of the input data...
        ...and is a numpy.ndarray with shape (1, m)
        A: contains the activated output of the neuron of each example...
        ...and is a numpy.ndarray with shape (1, m)
        m: the number of examples
        Σ: the sum over all examples
        To avoid division by zero errors, use 1.0000001 - A instead of 1 - A
        Returns the cost

        Cost function of logistic regression:
        cost = -1/m * Σ [Y * log(A) + (1 - Y) * log(1 - A)]
        """
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        X: numpy.ndarray with shape (nx, m)
            that contains the input data
        Y: numpy.ndarray with shape (1, m)
            that contains the correct labels of the input data
        Returns the neuron’s prediction
            and the cost of the network, respectively
        """
        # Generate predictions
        A = self.forward_prop(X)
        # Calculate cost
        cost = self.cost(Y, A)
        # Apply threshold to generate binary predictions
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        X: numpy.ndarray with shape (nx, m)
            that contains the input data
        Y: numpy.ndarray with shape (1, m)
            that contains the correct labels of the input data
        A: numpy.ndarray with shape (1, m)
            containing the activated output of the neuron of each example
        alpha: the learning rate
        Updates the private attributes __W and __b
        """
        m = X.shape[1]
        dZ = A - Y
        dW = np.dot(dZ, X.T) / m
        db = np.sum(dZ) / m
        self.__W = self.__W - alpha * dW.T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network
            by updating the private attributes __A, __W, and __b
        X: numpy.ndarray of shape
            (nx, m) that contains the input data
        Y: numpy.ndarray of shape (1, m)
            that contains the correct labels for the input data
        iterations: number of iterations to train the network
        alpha: learning rate
        verbose: boolean that defines if the cost
            should be printed every step
        graph: boolean that defines if the cost
            should be plotted every step
        step: number of iterations to print or plot the cost
        """
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')

        # Check if alpha is a positive float
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        # Check if step is a positive integer and less than iterations
        if verbose is True or graph is True:
            if type(step) != int:
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        # Initialize costs list to keep track of cost for each step iterations
        costs = []

        # Loop over iterations
        for i in range(iterations + 1):
            # Forward propagate to get the activation 'A'
            A = self.forward_prop(X)

            # If verbose, print the cost every step iterations
            if verbose and i % step == 0:
                print("Cost after {} iterations: {}".
                      format(i, self.cost(Y, A)))

            # If not the last iteration,
            # perform gradient descent to update weights and biases
            if i != iterations:
                self.gradient_descent(X, Y, A, alpha)

            # If graph, append the cost to costs list every step iterations
            if graph and i % step == 0:
                costs.append(self.cost(Y, self.A))

        # Plot the cost over iterations if graph is True
        if graph:
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        # Return the evaluation of the training data
        return self.evaluate(X, Y)
