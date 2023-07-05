#!/usr/bin/env python3
"""performs a convolution on images with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    images is a numpy.ndarray with shape (m, h, w, c)
        containing multiple images
        m: the number of images
        h: the height in pixels of the images
        w: the width in pixels of the images
        c: the number of channels in the image
    kernel is a numpy.ndarray with shape (kh, kw, c)
        containing the kernel for the convolution
        kh: the height of the kernel
        kw: the width of the kernel
        c: the number of channels in the image
    if necessary, the image should be padded with 0’s
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
            ph: the padding for the height of the image
            pw: the padding for the width of the image
    the image should be padded with 0’s
    stride is a tuple of (sh, sw)
        sh: the stride for the height of the image
        sw: the stride for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    c = kernel.shape[2]
    sh, sw = stride

    if padding == 'same':
        ph = ((((h - 1) * sh) + kh - h) // 2) + 1
        pw = ((((w - 1) * sw) + kw - w) // 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    elif type(padding) == tuple:
        ph, pw = padding

    convolvedW = ((w + (2 * pw) - kw) // sw) + 1
    convolvedH = ((h + (2 * ph) - kh) // sh) + 1
    padded_images = np.pad(images, ((0, 0),
                                    (ph, ph),
                                    (pw, pw),
                                    (0, 0)),
                           'constant')
    convolvedMatrix = np.zeros((m, convolvedH, convolvedW))

    for i in range(convolvedW):
        for j in range(convolvedH):
            shredder = padded_images[:, sh*j:sh*j + kh, sw*i:sw*i + kw, :]
            convolvedMatrix[:, j, i] = np.tensordot(shredder, kernel, axes=3)
    return convolvedMatrix
