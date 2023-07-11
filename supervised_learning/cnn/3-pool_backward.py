"""that performs back propagation over a pooling
layer of a neural network"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new)
        ...containing the partial derivatives with respect
        ...to the output of the pooling layer
        m: the number of examples
        h_new: the height of the output
        w_new: the width of the output
        c: the number of channels
    A_prev a numpy.ndarray of shape (m, h_prev, w_prev, c)
        ...containing the output of the previous layer
        h_prev: the height of the previous layer
        w_prev: the width of the previous layer
    kernel_shape: a tuple of (kh, kw)
        ...containing the size of the kernel for the pooling
        kh: the kernel height
        kw: the kernel width
    stride: a tuple of (sh, sw)
        ...containing the strides for the pooling
    sh: the stride for the height
    sw: the stride for the width
    mode: a string containing either max or avg,
        ...indicating whether to perform maximum or average pooling
    Returns: the partial derivatives w/ respect to the previous layer (dA_prev)
    """
    m = dA.shape[0]
    h_new = dA.shape[1]
    w_new = dA.shape[2]
    c_new = dA.shape[3]

    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c = A_prev.shape[3]

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    dA_prev = np.zeros_like(A_prev)
    for batch_index in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    if mode == 'avg':
                        average_dA = dA[batch_index, h, w, ch] / kh / kw
                        dA_prev[batch_index, sh*h:sh*h+kh,
                                 sw*w:sw*w+kw, ch] += (
                                    np.ones((kh, kw)) * average_dA
                                    )
                    if mode == 'max':
                        a_prev_slice = A_prev[
                            batch_index, sh*h:sh*h+kh, sw*w:sw*w+kw, ch
                            ]
                        max_mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[batch_index, sh*h:sh*h+kh,
                                sw*w:sw*w+kw, ch] += (
                                    max_mask * dA[batch_index, h, w, ch]
                                    )
    return dA_prev
