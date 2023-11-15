#!/usr/bin/env python3
"""
create class BidirectionalCell that represents a bidirectional cell of an RNN
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    performs forward propagation for a bidirectional RNN
    bi_cell is an instance of BidirectionalCell that will be used for the
        forward propagation
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    h_0 is the initial hidden state in the forward direction, given as a
        numpy.ndarray of shape (m, h)
        h is the dimensionality of the hidden state
    h_t is the initial hidden state in the backward direction, given as a
        numpy.ndarray of shape (m, h)
    Returns: H, Y
        H is a numpy.ndarray containing all of the concatenated hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
