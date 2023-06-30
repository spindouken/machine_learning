#!/usr/bin/env python3
"""performs a same convolution on grayscale images"""


def convolve_grayscale_same(images, kernel):
    """
    images is a numpy.ndarray with shape (m, h, w) containing multiple grayscale images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the convolution
    kh is the height of the kernel
    kw is the width of the kernel
    if necessary, the image should be padded with 0’s
    You are only allowed to use two for loops; any other loops of any kind are not allowed
    Returns: a numpy.ndarray containing the convolved images
    """
    
