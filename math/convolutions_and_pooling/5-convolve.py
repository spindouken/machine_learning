#!/usr/bin/env python3
"""performs a convolution on images using multiple kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """

    images is a numpy.ndarray with shape (m, h, w, c)
        containing multiple images
        m: the number of images
        h: the height in pixels of the images
        w: the width in pixels of the images
        c: the number of channels in the image
    kernel is a numpy.ndarray with shape (kh, kw, c, nc)
        containing the kernels for the convolution
        kh: the height of the kernel
        kw: the width of the kernel
        c: the number of channels in the image
        nc: number of kernels
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
    kh = kernels.shape[0]
    kw = kernels.shape[1]
    kc = kernels.shape[2]
    nc = kernels.shape[3]
    sh, sw = stride

    if padding == 'same':
        ph = ((((h - 1) * sh) + kh - h) // 2) + 1
        pw = ((((w - 1) * sw) + kw - w) // 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    elif type(padding) == tuple:
        ph, pw = padding

    convolvedW = ((w - kw + (2 * pw)) // sw) + 1
    convolvedH = ((h - kh + (2 * ph)) // sh) + 1
    padded_images = np.pad(images, ((0, 0),
                                    (ph, ph),
                                    (pw, pw),
                                    (0, 0)),
                           'constant')
    convolvedMatrix = np.zeros((m, convolvedH, convolvedW, nc))

    for i in range(nc):
        for w in range(convolvedW):
            for h in range(convolvedH):
                shredder = padded_images[:, sh*h:sh*h + kh, sw*w:sw*w + kw]
                convolvedMatrix[:, h, w, i] = np.tensordot(shredder,
                                                           kernels[:, :, :, i],
                                                           axes=3)
    return convolvedMatrix
