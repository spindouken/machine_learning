#!/usr/bin/env python3
"""performs a convolution on grayscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
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
    padding is a tuple of (ph, pw)
        ph: the padding for the height of the image
        pw: the padding for the width of the image
        the image should be padded with 0’s
    Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph, pw = padding

    # padd images with zeros
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    convolvedW = w - kw + 1 + (2 * pw)
    convolvedH = h - kh + 1 + (2 * ph)
    convolvedMatrix = np.zeros((m, convolvedH, convolvedW))

    for x in range(convolvedW):
        for z in range(convolvedH):
            shredder = padded_images[:, z:z + kh, x:x + kw]
            convolvedMatrix[:, z, x] = np.tensordot(shredder, kernel, axes=2)
    return convolvedMatrix
