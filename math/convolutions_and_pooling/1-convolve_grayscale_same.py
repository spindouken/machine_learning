#!/usr/bin/env python3
"""performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
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
    if necessary, the image should be padded with 0’s
    Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    # calculate padding
    padH = kh // 2
    padW = kw // 2

    # pad dem images
    padded_images = np.pad(
        images, ((0, 0), (padH, padH), (padW, padW))
        )

    convolvedMatrix = np.zeros((m, h, w))
    for x in range(w):
        for y in range(h):
            shredder = padded_images[:, y:y + kh, x:x + kw]
            if shredder.shape[1:] == kernel.shape:
                convolvedMatrix[:, y, x] = np.tensordot(shredder,
                                                        kernel,
                                                        axes=2)
    return convolvedMatrix
