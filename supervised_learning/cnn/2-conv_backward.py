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

    m = dZ.shape[0]
    h_new = dZ.shape[1]
    w_new = dZ.shape[2]
    c_new = dZ.shape[3]

    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kh = W.shape[0]
    kw = W.shape[1]

    sh = stride[0]
    sw = stride[1]

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'valid':
        pad_h, pad_w = 0, 0
    elif padding == 'same':
        pad_h = (((h_prev - 1) * sh) + kh - h_prev) // 2 + 1
        pad_w = (((w_prev - 1) * sw) + kw - w_prev) // 2 + 1

    A_prev = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                    mode='constant', constant_values=0)
    dA = np.pad(dA_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                mode='constant', constant_values=0)

    for image in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    kernel = W[:, :, :, c]
                    dz = dZ[image, h, w, c]
                    mat = A_prev[image, sh*h:sh*h+kh, sw*w:sw*w+kw, :]
                    dW[:, :, :, c] += mat * dz
                    dA[image, sh*h:sh*h+kh,
                       sw*w:sw*w+kw, :] += np.multiply(kernel, dz)
    if padding == 'same':
        dA = dA[:, pad_h: -pad_h, pad_w: -pad_w, :]
    return dA, dW, db
