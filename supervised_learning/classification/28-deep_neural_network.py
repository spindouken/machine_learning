#!/usr/bin/env python3
"""are you winning at neural networks son?"""
import pickle
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """
    defines a deep neural network performing binary classification
    """
    def __init__(self, nx, layers, activation='sig'):
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
        activation: The activation function to be used in the hidden layers.
            'sig' for sigmoid function and 'tanh' for tanh function.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
        for i in range(self.__L):
            self.__weights['W' + str(i+1)] = np.random.randn(
                layers[i], nx) * np.sqrt(2/nx)
            self.__weights['b' + str(i+1)] = np.zeros((layers[i], 1))
            nx = layers[i]

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @property
    def activation(self):
        return self.__activation

    def forward_prop(self, X):
        """Forward Propagation"""
        self.__cache.update({"A0": X})
        for layer in range(1, self.L+1):
            weights_current_layer = self.weights["W{}".format(layer)]
            activations_previous_layer = self.cache["A{}".format(layer-1)]
            biases_current_layer = self.weights["b{}".format(layer)]
            z = np.matmul(weights_current_layer,
                          activations_previous_layer) + biases_current_layer
            if layer == self.L:
                t = np.exp(z)
                self.__cache["A{}".format(layer)] = t/np.sum(t, axis=0)
            else:
                if self.__activation == 'sig':
                    self.__cache["A{}".format(layer)] = 1 / (1 + np.exp(-z))
                elif self.__activation == 'tanh':
                    self.__cache["A{}".format(layer)] = np.tanh(z)
        return self.cache["A{}".format(self.L)], self.cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Y: a numpy.ndarray with shape (1, m)
            that contains the correct labels of the input data
        A: a numpy.ndarray with shape (1, m)
            containing the activated output of the neuron of each example
        Returns the cost of the model as a float
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions
        X: a numpy.ndarray with shape (nx, m) that contains the input data
        Y: a numpy.ndarray with shape (1, m)
            that contains the correct labels of the input data
        Returns the neuron's prediction and the cost of the network
        """
        # Compute the forward propagation and get the final activation value
        final_activation = self.forward_prop(X)[0]
        # Get the indices of the maximum activation value for each example
        max_activation_indices = np.argmax(self.cache["A3"], axis=0)
        max_activation_indices.reshape(max_activation_indices.size, 1)
        # Create an array of indices for each example
        example_indices = np.arange(final_activation.shape[1])
        example_indices.reshape(example_indices.size, 1)
        # Create a matrix of zeros with the same shape as the final activation
        hard_max = np.zeros_like(final_activation)
        # Set the maximum activation value for each example to 1
        hard_max[max_activation_indices, example_indices] = 1
        # Return the predictions (as a binary matrix) and the cost
        return (hard_max.astype(int),
                self.cost(Y, self.cache["A{}".format(self.L)]))

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        Y: a numpy.ndarray with shape (1, m)
            that contains the correct labels of the input data
        cache: a dictionary containing all intermediary values of the network
        alpha: the learning rate
        Updates the private attribute __weights
        """
        m = Y.shape[1]
        # Loop over each layer in the network in reverse order
        for layer in reversed(range(1, self.__L + 1)):
            # Retrieve the activations for the current and previous layer
            activations_current_layer = cache['A' + str(layer)]
            activations_previous_layer = cache['A' + str(layer - 1)]
            # Retrieve the weights for the current layer
            weights_current_layer = self.__weights['W' + str(layer)]
            # Compute the gradient of the activation function
            if layer == self.__L:
                # For the output layer, the gradient is
                #   the difference between the activations and the labels
                gradient = activations_current_layer - Y
            else:
                # For the hidden layers, gradient is the product of
                #   the derivative of activation func and prev gradient
                if self.__activation == 'sig':
                    gradient = gradient_previous_layer * \
                               self.sigmoid_derivative(
                                activations_current_layer
                                )
                elif self.__activation == 'tanh':
                    gradient = gradient_previous_layer * \
                               (1 - activations_current_layer ** 2)
            # Compute the gradients of the weights and biases
            gradient_weights = np.dot(
                gradient, activations_previous_layer.T
                ) / m
            gradient_biases = np.sum(gradient, axis=1, keepdims=True) / m
            # If not at the first layer, compute gradient for previous layer
            if layer > 1:
                gradient_previous_layer = np.dot(
                    weights_current_layer.T, gradient
                    )
            # Update the weights and biases
            self.__weights['W' + str(layer)] -= alpha * gradient_weights
            self.__weights['b' + str(layer)] -= alpha * gradient_biases

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the deep neural network
        X: a numpy.ndarray with shape
            (nx, m) that contains the input data
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
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                costs.append(cost)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))

        if graph is True:
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object
        Returns the loaded pickle, or None if filename doesn’t exist
        """
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None

    @staticmethod
    def sigmoid(Z):
        """Sigmoid Activation"""
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def sigmoid_derivative(A):
        """Sigmoid Derivative"""
        return A * (1 - A)
