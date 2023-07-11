#!/usr/bin/env python3
"""
Module to perform a forward pass over a convolutional layer in a neural network
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network

    parameters:
        - A_prev [numpy.ndarray of shape (m, h_prev, w_prev, c_prev)]:
            contains output of previous layer
            * m: number of examples
            * h_prev: height of previous layer
            * w_prev: width of previous layer
            * c_prev: number of channels in previous layer
        - W [numpy.ndarray of shape (kh, kw, c_prev, c_new)]:
            contains kernels for convolution
            * kh: filter height
            * kw: filter width
            * c_prev: number of channels in previous layer
            * c_new: number of channels in output
        - b [numpy.ndarray of shape (1, 1, 1, c_new)]:
            contains biases applied to convolution
        - activation [function]:
            activation function applied to convolution
        - padding [str]:
            either 'same' or 'valid', indicating type of padding used
        - stride [tuple of (sh, sw)]:
            contains strides for convolution
            * sh: stride for height
            * sw: stride for width

    returns:
        output of convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph = pw = 0

    h_new = (h_prev + 2 * ph - kh) // sh + 1
    w_new = (w_prev + 2 * pw - kw) // sw + 1

    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    Z = np.zeros((m, h_new, w_new, c_new))

    for c in range(c_new):
        for h in range(h_new):
            for w in range(w_new):
                start_h = h * sh
                start_w = w * sw
                end_h = start_h + kh
                end_w = start_w + kw

                a_slice = A_prev_pad[:, start_h:end_h, start_w:end_w, :]
                weights = W[:, :, :, c]
                biases = b[0, 0, 0, c]

                Z[:, h, w, c] = np.sum(
                    a_slice * weights, axis=(1, 2, 3)) + biases

    A = activation(Z)
    return A
