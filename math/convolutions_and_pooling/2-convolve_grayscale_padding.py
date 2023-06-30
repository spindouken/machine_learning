#!/usr/bin/env python3
"""performs a convolution on grayscale images with custom padding"""


def convolve_grayscale_padding(images, kernel, padding):
    """
    images is a numpy.ndarray with shape (m, h, w) containing multiple grayscale images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the convolution
    kh is the height of the kernel
    kw is the width of the kernel
    padding is a tuple of (ph, pw)
    ph is the padding for the height of the image
    pw is the padding for the width of the image
    the image should be padded with 0’s
    You are only allowed to use two for loops; any other loops of any kind are not allowed
    Returns: a numpy.ndarray containing the convolved images
    """

    