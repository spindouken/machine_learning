#!/usr/bin/env python3
"""
performs forward propagation for a simple RNN
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    rnn_cell is an instance of RNNCell that will
        be used for the forward propagation
    X is the data to be used, given as a numpy.ndarray
        of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    h_0 is the initial hidden state, given as a numpy.ndarray
        of shape (m, h)
        h is the dimensionality of the hidden state
    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    _, h = h_0.shape

    H = np.zeros((t + 1, m, h))
    H[0] = h_0  # initial hidden state

    _, o = rnn_cell.Wy.shape
    Y = np.zeros((t, m, o))

    for timeStep in range(t):
        # update hidden state and output using RNNCell's forward method
        H_out, Y_out = rnn_cell.forward(H[timeStep], X[timeStep])
        H[timeStep + 1] = H_out
        Y[timeStep] = Y_out

    return H, Y
