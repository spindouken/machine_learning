#!/usr/bin/env python3
"""
performs forward propagation for a deep RNN
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    performs forward propagation for a deep RNN
    rnn_cells is a list of RNNCell instances of length l
        that will be used for the forward propagation
        l is the number of layers
    X is the data to be used, given as a numpy.ndarray
        of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    h_0 is the initial hidden state, given as a numpy.ndarray
        of shape (l, m, h)
        h is the dimensionality of the hidden state
    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the Y
    """
    # extract dimensions from the inputs
    numTimeSteps, batchSize, _ = X.shape
    numLayers = len(rnn_cells)
    hiddenSize = h_0.shape[2]

    # initialize hidden states, including the initial state
    #   extra time step added for initial hidden state
    H = np.zeros((numTimeSteps + 1, numLayers, batchSize, hiddenSize))
    H[0] = h_0

    # preparing to store outputs
    outputSize = rnn_cells[-1].Wy.shape[1]  # determine out size of last layer
    # initialize Y matrix (for output storage) w/ zeros using extracted dims
    Y = np.zeros((numTimeSteps, batchSize, outputSize))

    # loop through each time step
    for timeStep in range(numTimeSteps):
        # start with input data for the first layer
        currentInput = X[timeStep]

        # processing through each layer
        for layer in range(numLayers):
            # get RNN cell and previous hidden state
            cell = rnn_cells[layer]
            # h_prev holds the state from the previous time step for this layer
            h_prev = H[timeStep, layer]

            # forward step on the current cell, computing next hidden state
            #   and output for the current time step
            # y = output of the current cell
            h_next, y = cell.forward(h_prev, currentInput)

            # store the computed next hidden state
            #   this will be used as the previous state in the next time step
            H[timeStep + 1, layer] = h_next

            # update input for the next layer with the current hidden state
            currentInput = h_next

            # update output if it's the last layer
            if layer == numLayers - 1:
                # store the output of the current cell (y) in the Y matrix
                Y[timeStep] = y

    # return all hidden states (excluding the initial state) and outputs
    return H, Y
