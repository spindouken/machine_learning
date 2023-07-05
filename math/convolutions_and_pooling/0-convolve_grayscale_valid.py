#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    images is a numpy.ndarray with shape (m, h, w)
        containing multiple grayscale images
        m: the number of images
        h: the height in pixels of the images
        w: the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
        containing the kernel for the convolution
        kh: the height of the kernel
        kw: the width of the kernel
    Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    convolvedW = w - kw + 1
    convolvedH = h - kh + 1
    convolvedMatrix = np.zeros((m, convolvedH, convolvedW))

    for x in range(convolvedW):
        for y in range(convolvedH):
            shredder = images[:, y:y + kh, x:x + kw]
            convolvedMatrix[:, y, x] = np.tensordot(shredder,
                                                    kernel,
                                                    axes=2)
    return convolvedMatrix
