#!/usr/bin/env python3
"""performs forward propagation over a pooling
layer of a neural network"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        ...containing the output of the previous layer
        m: number of examples
        h_prev: height of the previous layer
        w_prev: width of the previous layer
        c_prev: number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw)
        ....containing the size of the kernel for the pooling
        kh: kernel height
        kw: kernel width
    stride is a tuple of (sh, sw)
        ...containing the strides for the pooling
        sh: stride for the height
        sw: stride for the width
    mode: string containing either max or avg,
        ...indicating whether to perform maximum or average pooling
    Returns: the output of the pooling layer
    """
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    if mode == 'max':
        op = np.max
    elif mode == 'avg':
        op = np.mean

    h_pool = (h_prev - kh) // sh + 1
    w_pool = (w_prev - kw) // sw + 1
    poolOut = np.zeros((m, h_pool, w_pool, c_prev))

    for h in range(h_pool):
        for w in range(w_pool):
            poolOut[:, h, w, :] = op(A_prev[:, sh*h:sh*h+kh, sw*w:sw*w+kw],
                                     axis=(1, 2))
    return poolOut
