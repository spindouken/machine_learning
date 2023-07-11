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
    padding: string that is either same or valid, indicating the type of padding used
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
    c_prev = W.shape[2]
    c_new = W.shape[3]

    # retrieve strides
    sh = stride[0]
    sw = stride[1]
    
    # calculate padding for 'same' and 'valid'
    if padding == "same":
        pad_h = max((h_prev - 1) * sh + kh - h_prev, 0) // 2
        pad_w = max((w_prev - 1) * sw + kw - w_prev, 0) // 2
    elif padding == "valid":
        pad_h = pad_w = 0

    # compute the dimensions of the CONV output volume
    h_new = (h_prev - kh + 2 * pad_h) // sh + 1
    w_new = (w_prev - kw + 2 * pad_w) // sw + 1

    # initialize the output volume Z with zeros
    Z = np.zeros((m, h_new, w_new, c_new))

    # create A_prev_pad by padding A_prev
    A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    # loop over the vertical (height), horizontal (width) and channel axes of the output volume
    for i in range(m):  # loop over the batch size
        for h in range(h_new):  # loop over the vertical axis
            for w in range(w_new):  # loop over the horizontal axis
                for c in range(c_new):  # loop over the channels

                    # find the corners of the current slice
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    # use the corners to define the slice from a_prev_pad
                    a_slice = A_prev_pad[i, h_start:h_end, w_start:w_end, :]

                    # convolve the slice with the correct filter and bias to get the output neuron
                    Z[i, h, w, c] = np.sum(a_slice * W[:, :, :, c]) + b[0, 0, 0, c]

    # Applying the activation function
    A = activation(Z)

    return A
