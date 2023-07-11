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
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kh = W.shape[0]
    kw = W.shape[1]
    c_prev = W.shape[2]
    c_new = W.shape[3]

    sh = stride[0]
    sw = stride[1]
    
    # calculate padding for 'same' and 'valid'
    if padding == "same":
        pad_h = (((h_prev - 1) * sh) + kh - h_prev) // 2
        pad_w = (((w_prev - 1) * sw) + kw - w_prev) // 2

    elif padding == 'valid':
        pad_h = pad_w = 0

    # calculate output dimensions
    h_out = (h_prev - kh + 2 * pad_h) // sh + 1
    w_out = (w_prev - kw + 2 * pad_w) // sw + 1

    # pad the input
    A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    Z = np.zeros((m, h_out, w_out, c_new))

    # loop over the batch of training examples
    for i in range(m):
        # loop over vertical axis of the output volume
        for h in range(h_out):
            # loop over horizontal axis of the output volume
            for w in range(w_out):
                # calculate slice parameters
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw
                a_slice_prev = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]
                # loop over channels of the output volume
                for c in range(c_new):
                    # perform the convolution
                    Z[i, h, w, c] = np.sum(a_slice_prev * W[:, :, :, c]) + b[0, 0, 0, c]

    # Apply the activation function
    A = activation(Z)
    return A
