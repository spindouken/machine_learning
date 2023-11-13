#!/usr/bin/env python3
"""
create the class LSTMCell that represents an LSTM unit
"""
import numpy as np


class LSTMCell:
    """
    represents an LSTM unit
    """
    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wf, Wu, Wc, Wo, Wy,
                bf, bu, bc, bo, by
            Wf and bf are for the forget gate
            Wu and bu are for the update gate
            Wc and bc are for the intermediate cell state
            Wo and bo are for the output gate
            Wy and by are for the outputs
        The weights should be initialized using a random normal distribution
            in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """
        self.Wf = np.random.randn(i + h, h)
        self.bf = np.zeros((1, h))
        self.Wu = np.random.randn(i + h, h)
        self.bu = np.zeros((1, h))
        self.Wc = np.random.randn(i + h, h)
        self.bc = np.zeros((1, h))
        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        X_t is a numpy.ndarray of shape (m, i) that contains the data input
            for the cell
            m is the batch size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
            hidden state
        c_prev is a numpy.ndarray of shape (m, h) containing the previous
            cell state
        The output of the cell should use a softmax activation function
        Returns: h_next, c_next, y
            h_next is the next hidden state
            c_next is the next cell state
            y is the output of the cell
        """
        # using concatenate to merge the previous hidden state w/ new input
        #   this is crucial for maintaining the temporal aspect of the LSTM
        combined = np.concatenate((h_prev, x_t), axis=1)

        # forget gate: decides which information to discard from the cell state
        #   using sigmoid as it outputs values between 0 and 1,ideal for gating
        forgetGateRaw = combined @ self.Wf + self.bf
        forgetGateActivation = self.sigmoid(forgetGateRaw)

        # input gate: updates the cell state with new information
        #   sigmoid used for gating - binary decision making
        inputGateRaw = combined @ self.Wu + self.bu
        inputGateActivation = self.sigmoid(inputGateRaw)

        # candidate cell state: combines old and new information
        #   tanh used to regulate the values to be between -1 and 1
        candidateCellRaw = combined @ self.Wc + self.bc
        candidateCellState = np.tanh(candidateCellRaw)

        # next cell state: combination of old state and new candidate state
        c_next = (
            forgetGateActivation * c_prev
            + inputGateActivation * candidateCellState
        )

        # output gate: decides what parts of the cell state make it to output
        #   sigmoid is used for gating - binary decision making
        outputGateRaw = combined @ self.Wo + self.bo
        outputGateActivation = self.sigmoid(outputGateRaw)

        # next hidden state: refined version of the cell state
        h_next = outputGateActivation * np.tanh(c_next)

        # final output: convert hidden state to the desired output format
        #   softmax used to normalize output into a probability distribution
        outputRaw = h_next @ self.Wy + self.by
        # y = softmaxed output of the cell
        y = self.softmax(outputRaw)

        return h_next, c_next, y

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """softmax activation function"""
        max = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - max)
        sum = np.sum(e_x, axis=1, keepdims=True)
        f_x = e_x / sum
        return f_x
