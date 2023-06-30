#!/usr/bin/env python3
"""performs pooling on images"""


def pool(images, kernel_shape, stride, mode='max'):
    """
    images is a numpy.ndarray with shape (m, h, w, c) containing multiple images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    c is the number of channels in the image
    kernel_shape is a tuple of (kh, kw) containing the kernel shape for the pooling
    kh is the height of the kernel
    kw is the width of the kernel
    stride is a tuple of (sh, sw)
    sh is the stride for the height of the image
    sw is the stride for the width of the image
    mode indicates the type of pooling
    max indicates max pooling
    avg indicates average pooling
    You are only allowed to use two for loops; any other loops of any kind are not allowed
    Returns: a numpy.ndarray containing the pooled images
    """

