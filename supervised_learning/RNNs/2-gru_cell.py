#!/usr/bin/env python3
"""
creates a class GRUCell that represents a gated recurrent unit
"""
import numpy as np


class GRUCell:
    """
    represents a gated recurrent unit
    """

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
            Wz and bz are for the update gate
            Wr and br are for the reset gate
            Wh and bh are for the intermediate hidden state
            Wy and by are for the output
        The weights should be initialized using a random normal distribution
            in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step
        x_t is a numpy.ndarray of shape (m, i) that contains the data input
            for the cell
            m is the batch size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
            hidden state
        The output of the cell should use a softmax activation function
        Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        # update Gate
        #   combines input and previous hidden state to update cell's memory
        updateInput = x_t @ self.Wz[h_prev.shape[1]:, :]
        updateHidden = h_prev @ self.Wz[:h_prev.shape[1], :]
        updateGate = self.sigmoid(updateInput + updateHidden + self.bz)

        # reset gate
        #   determines how much past information to forget
        resetInput = x_t @ self.Wr[h_prev.shape[1]:, :]
        resetHidden = h_prev @ self.Wr[: h_prev.shape[1], :]
        resetGate = self.sigmoid(resetInput + resetHidden + self.br)

        # candidate hidden state
        #   combination of new input and the previous hidden state
        candidateInput = x_t @ self.Wh[h_prev.shape[1]:, :]
        candidateHidden = (h_prev * resetGate) @ self.Wh[: h_prev.shape[1], :]
        candidateState = np.tanh(candidateInput + candidateHidden + self.bh)

        # calculate next hidden state
        h_next = (h_prev * (1 - updateGate)) + (updateGate * candidateState)

        # output
        #   apply final transformation to the hidden state to get the output
        y = self.softmax(h_next @ self.Wy + self.by)

        return h_next, y

    def softmax(self, x):
        """softmax activation function"""
        max = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - max)
        sum = np.sum(e_x, axis=1, keepdims=True)
        f_x = e_x / sum
        return f_x

    def sigmoid(self, x):
        """sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
