#!/usr/bin/env python3
"""performs a convolution on images with channels"""


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    images is a numpy.ndarray with shape (m, h, w, c) containing multiple images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    c is the number of channels in the image
    kernel is a numpy.ndarray with shape (kh, kw, c) containing the kernel for the convolution
    kh is the height of the kernel
    kw is the width of the kernel
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
    if ‘same’, performs a same convolution
    if ‘valid’, performs a valid convolution
    if a tuple:
    ph is the padding for the height of the image
    pw is the padding for the width of the image
    the image should be padded with 0’s
    stride is a tuple of (sh, sw)
    sh is the stride for the height of the image
    sw is the stride for the width of the image
    You are only allowed to use two for loops; any other loops of any kind are not allowed
    Returns: a numpy.ndarray containing the convolved images
    """
    