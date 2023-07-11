#!/usr/bin/env python3
"""performs forward propagation over
a convolutional layer of a neural network"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        ...containing the output of the previous layer
        m: the number of examples
        h_prev:  height of the previous layer
        w_prev:  width of the previous layer
        c_prev: number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
        ...containing the kernels for the convolution
        kh: filter height
        kw: filter width
        c_prev: number of channels in the previous layer
        c_new: number of channels in the output
    b is a numpy.ndarray of shape (1, 1, 1, c_new)
        ...containing the biases applied to the convolution
    activation: an activation function applied to the convolution
    padding: string that is either same or valid, indic type of padding used
    stride: a tuple of (sh, sw) containing the strides for the convolution
        sh: stride for the height
        sw: stride for the width
    Returns: the output of the convolutional layer
    """
    # retrieve dimensions from A_prev's shape
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    # retrieve dimensions from W's shape
    kh = W.shape[0]
    kw = W.shape[1]
    _ = W.shape[2]
    c_new = W.shape[3]

    # retrieve strides
    sh = stride[0]
    sw = stride[1]

    # initialize padding for height and width
    ph, pw = 0, 0

    # calculate padding for height and width if 'same'
    if padding == "same":
        ph = ((h_prev - 1) * sh - h_prev + kh) // 2
        pw = ((w_prev - 1) * sw - w_prev + kw) // 2

    # pad the previous layer
    padded_A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))

    # calculate dimensions for output
    h_out = (h_prev - kh + 2 * ph) // sh + 1
    w_out = (w_prev - kw + 2 * pw) // sw + 1

    # initialize output
    Z = np.zeros((m, h_out, w_out, c_new))

    # convolution operation
    for i in range(m):  # loop over the batch of training examples
        for h in range(h_out):  # loop over vertical axis of output volume
            for w in range(w_out):  # loop over horiz axis of output volume
                for c in range(c_new):  # loop over channels (#filters) of ov
                    # find the corners of the current "slice"
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    # use corners to define current slice on ith
                    # training example of padded_A_prev
                    a_slice_prev = padded_A_prev[
                        i, vert_start:vert_end, horiz_start:horiz_end, :
                        ]

                    # convolve the (3D) slice w/ correct filter W and bias b,
                    # to get the output neuron
                    Z[i, h, w, c] = np.sum(
                        a_slice_prev * W[..., c]
                        ) + b[..., c]

    # apply activation function
    A = activation(Z)
    return A
