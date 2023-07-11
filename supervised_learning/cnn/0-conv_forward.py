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

    # Compute padding dimensions
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:  # padding == "valid"
        ph = pw = 0

    # Compute the dimensions of the CONV output volume
    h_new = (h_prev + 2 * ph - kh) // sh + 1
    w_new = (w_prev + 2 * pw - kw) // sw + 1

    # Initialize the output volume Z with zeros
    Z = np.zeros((m, h_new, w_new, c_new))

    # Create A_prev_pad by padding A_prev
    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    # loop over the vertical (h), then horizontal (w), then over channels (c)
    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                # Find the corners of the current slice
                start_h = i * sh
                start_w = j * sw
                end_h = start_h + kh
                end_w = start_w + kw

                # Use the corners to define the slice from A_prev_pad
                a_slice_prev = A_prev_pad[:, start_h:end_h, start_w:end_w, :]

                # Convolve the (3D) slice with the correct filter W and bias b,
                # to get back one output neuron
                weights = W[:, :, :, k]
                biases = b[0, 0, 0, k]
                Z[:, i, j, k] = np.sum(
                    a_slice_prev * weights, axis=(1, 2, 3)) + biases

    # Apply the activation function
    A = activation(Z)
    return A
