#!/usr/bin/env python3
"""
creates a class RNNCell that represents a cell of a simple RNN
"""
import numpy as np


def softmax(X):
    """
    softmax function
    """
    exponentiatedX = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exponentiatedX / exponentiatedX.sum(axis=1, keepdims=True)


class RNNCell:
    """
    represents a cell of a simple RNN
    """
    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wh, Wy, bh, by that represent
            the weights and biases of the cell
                Wh and bh are for the concatenated hidden state and input data
                Wy and by are for the output
        The weights should be initialized using a random normal distribution
            in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """
        # i + h = h_prev + x_t
        #     = total number of features input in RNNCell at each time step
        #     (includes input features and features from prev hidden state)
        #     (in a simple RNN, input at step
        #       is current data point concat w/ prev data point)
        # i + h rows and h columns
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros(shape=(1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """
        x_t is a numpy.ndarray of shape (m, i) that contains the data input
            for the cell
                m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
            hidden state
        The output of the cell should use a softmax activation function
        Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        # concatenate previous hidden state and current input
        # why? an RNN cell considers both the new input
        #   and the past context when making predictions
        inputAndPrevState_combined = np.concatenate((h_prev, x_t), axis=1)

        # calc next hidden state by multiplying inputAndPrevState_combined by
        #   weight matrix for hidden state and adding bias
        #   use hyperbolic tangent (tanh) activation func to scale to [-1, 1]
        h_next = np.tanh(inputAndPrevState_combined @ self.Wh + self.bh)

        # use next hidden state calculation (h_next) to compute output of cell
        #   by multiplying h_next by weight matrix for output and adding bias
        unactivatedOutput = h_next @ self.Wy + self.by

        # softmax for classification (activated output)
        #   (returns a vector of probabilities summing up to 1)
        y = softmax(unactivatedOutput)

        # h_next used for subsequent time steps
        # y is output of the cell (current prediction)
        return h_next, y
