#!/usr/bin/env python3
"""
performs back propagation
over a convolutional layer of a neural network
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new)
        ...containing the partial derivatives with respect
        ...to the unactivated output of the convolutional layer
        m: the number of examples
        h_new: the height of the output
        w_new: the width of the output
        c_new: the number of channels in the output
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        ...containing the output of the previous layer
        h_prev: the height of the previous layer
        w_prev: the width of the previous layer
        c_prev: the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
        ...containing the kernels for the convolution
        kh: the filter height
        kw: the filter width
    b is a numpy.ndarray of shape (1, 1, 1, c_new)
        ...containing the biases applied to the convolution
    padding is a string that is either same or valid,
        ...indicating the type of padding used
    stride is a tuple of (sh, sw)
        ...containing the strides for the convolution
        sh: the stride for the height
        sw: the stride for the width
    Returns: the partial derivatives with respect
        to the previous layer (dA_prev), the kernels (dW),
        and the biases (db), respectively
    """
    # retrieve dimensions from dZ's shape
    m = dZ.shape[0]
    h_new = dZ.shape[1]
    w_new = dZ.shape[2]
    c_new = dZ.shape[3]

    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    # Retrieve dimensions from W's shape
    kh = W.shape[0]
    kw = W.shape[1]

    sh = stride[0]
    sw = stride[1]

    # initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, h_prev, w_prev, c_prev))
    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.zeros((1, 1, 1, c_new))

    # Compute padding dimensions
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:  # padding == "valid"
        ph = pw = 0

    # pad A_prev and dA_prev
    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    dA_prev_pad = np.pad(
        dA_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    # ;oop over the vertical (h), then horizontal (w), then over channels (c)
    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                # find the corners of the current slice
                start_h = i * sh
                start_w = j * sw
                end_h = start_h + kh
                end_w = start_w + kw

                # update gradients for the window and
                # the filter's params using the code formulas
                a_slice = A_prev_pad[:, start_h:end_h, start_w:end_w, :]

                # update gradients of the slice of A_prev_pad (dA_prev_pad)
                dA_prev_pad[
                    :, start_h:end_h, start_w:end_w, :
                    ] += W[:, :, :, k] * dZ[
                        :, i, j, k, None, None, None
                        ]
                dW[:, :, :, k] += np.sum(
                    a_slice * dZ[:, i, j, k, None, None, None], axis=0)
                db[:, :, :, k] += np.sum(dZ[:, i, j, k])

    # set the ith training example's dA_prev to the unpaded da_prev_pad
    if padding == "same":
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
