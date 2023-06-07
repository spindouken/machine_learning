#!/usr/bin/env python3
"""are you winning at neural networks son?"""
import numpy as np
import matplotlib.pyplot as plt


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
            # Retrieve the weights and bias of current layer
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
            containing the activated output of the neuron of each example
        Returns the cost of the model as a float
        """
        # m is the number of examples
        m = Y.shape[1]

        # Calculate the loss of each training example
        #    using the formula of cross-entropy loss
        # The term 1.0000001 - A is used instead of 1 - A
        #    to avoid division by zero errors when A is exactly 1
        loss = - (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        # Calculate the cost as the average of the losses
        cost = np.sum(loss) / m

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions
        X: a numpy.ndarray with shape (nx, m) that contains the input data
        Y: a numpy.ndarray with shape (1, m)
            that contains the correct labels of the input data
        Returns the neuron's prediction and the cost of the network
        """
        # Use forward propagation to generate predictions
        A, _ = self.forward_prop(X)

        # Calculate the cost of these predictions
        cost = self.cost(Y, A)

        # Generate binary predictions: if the output of the network is >= 0.5,
        #    the predicted label is 1, otherwise it's 0
        predictions = np.where(A >= 0.5, 1, 0)

        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        Y: a numpy.ndarray with shape (1, m)
            that contains the correct labels of the input data
        cache: a dictionary containing all intermediary values of the network
        alpha: the learning rate
        Updates the private attribute __weights
        """
        # m is the number of examples
        m = Y.shape[1]

        # Start the backpropagation process from the last layer
        dZ = cache['A' + str(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            # Retrieve the data from the cache
            A_prev = cache['A' + str(i-1)]
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]

            # Compute the derivatives
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dZ = np.matmul(W.T, dZ) * A_prev * (1 - A_prev)

            # Update the weights and biases
            self.__weights['W' + str(i)] -= alpha * dW
            self.__weights['b' + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network
        X: a numpy.ndarray with shape (nx, m) that contains the input data
        Y: a numpy.ndarray with shape (1, m)
            that contains the correct labels of the input data
        iterations: the number of iterations to train over
        alpha: the learning rate
        verbose: boolean that defines whether or not
            to print information about the training
        graph: boolean that defines whether or not
            to graph information about the training
        step: the interval of printing information and updating the graph
        Updates the private attributes __weights and __cache
        Returns the evaluation of the training data
            after iterations of training have occurred
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # Initialize the list of costs of graphing
        costs = []

        # Train the network according to the specified number of iterations
        for i in range(iterations + 1):
            # Perform forward propagation
            A, _ = self.forward_prop(X)

            # Perform gradient descent
            self.gradient_descent(Y, self.__cache, alpha)

            # Calculate the cost
            cost = self.cost(Y, A)

            # Print the cost every `step` iterations
            if verbose is True and i % step == 0:
                print("Cost after {} iterations: {}".format(i, cost))

            # Save the cost every `step` iterations 4 graphing
            if graph is True and i % step == 0:
                costs.append(cost)

        # Graph the cost data
        if graph is True:
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        # Evaluate the training data after training
        return self.evaluate(X, Y)
