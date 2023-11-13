#!/usr/bin/env python3
"""
create the class BidirectionalCell that represents a bidirectional cell of an RNN
"""
import numpy as np


class BidirectionalCell:
    """
    represents a bidirectional cell of an RNN
    """

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden states
        o is the dimensionality of the outputs
        Creates the public instance attributes Whf, Whb, Wy, bhf, bhb, by
            that represent the weights and biases of the cell
            Whf and bhf are for the hidden states in the forward direction
            Whb and bhb are for the hidden states in the backward direction
            Wy and by are for the outputs
        The weights should be initialized using a random normal distribution
            in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """
        self.Whf = np.random.randn(h + i, h)  # weights for forward direction
        self.bhf = np.zeros((1, h))  # bias for forward direction
        self.Whb = np.random.randn(h + i, h)  # weights for backward direction
        self.bhb = np.zeros((1, h))  # bias for backward direction
        self.Wy = np.random.randn(2 * h, o)  # weights for outputs
        self.by = np.zeros((1, o))  # bias for outputs

    def forward(self, h_prev, x_t):
        """
        calculates the hidden state in the forward direction for one time step
        x_t is a numpy.ndarray of shape (m, i) that contains the data input
            for the cell
            m is the batch size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
            hidden state
        Returns: h_next, the next hidden state
        """
        # combine previous hidden state with input
        combined = np.concatenate((h_prev, x_t), axis=1)
        # apply activation function to the product of combined and Whf plus
        h_next = np.tanh(combined @ self.Whf + self.bhf)
        return h_next
