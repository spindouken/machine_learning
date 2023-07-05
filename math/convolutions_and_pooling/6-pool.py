#!/usr/bin/env python3
"""performs pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    images is a numpy.ndarray with shape (m, h, w, c)
        containing multiple images
        m: the number of images
        h: the height in pixels of the images
        w: the width in pixels of the images
        c: the number of channels in the image
    kernel_shape is a tuple of (kh, kw)
        containing the kernel shape for the pooling
        kh: the height of the kernel
        kw: the width of the kernel
    stride is a tuple of (sh, sw)
        sh: the stride for the height of the image
        sw: the stride for the width of the image
    mode indicates the type of pooling
        max: indicates max pooling
        avg: indicates average pooling
    Returns: a numpy.ndarray containing the pooled images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh, kw = kernel_shape
    sh, sw = stride

    if mode == "max":
        op = np.amax
    else:
        op = np.average

    ph, pw = 0, 0
    pooledW = (w - kw + (2 * pw)) // sw + 1
    pooledH = (h - kh + (2 * ph)) // sh + 1
    pooledMatrix = np.zeros((m, pooledH, pooledW, c))

    for i in range(pooledH):
        for j in range(pooledW):
            pooledMatrix[:, i, j, :] = op(
                images[:, sh*i: sh*i+kh, sw*j:sw*j+kw, :], axis=(1, 2)
                )
    return pooledMatrix
