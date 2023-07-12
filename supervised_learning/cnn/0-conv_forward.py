#!/usr/bin/env python3
"""performs forward propagation over
a convolutional layer of a neural network"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        ...containing the output of the previous layer
        m: the number of examples
        h_prev:  height of the previous layer
        w_prev:  width of the previous layer
        c_prev: number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
        ...containing the kernels for the convolution
        kh: filter height
        kw: filter width
        c_prev: number of channels in the previous layer
        c_new: number of channels in the output
    b is a numpy.ndarray of shape (1, 1, 1, c_new)
        ...containing the biases applied to the convolution
    activation: an activation function applied to the convolution
    padding: string that is either same or valid, indic type of padding used
    stride: a tuple of (sh, sw) containing the strides for the convolution
        sh: stride for the height
        sw: stride for the width
    Returns: the output of the convolutional layer
    """
    # retrieve dimensions from A_prev's shape
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    # retrieve dimensions from W's shape
    kh = W.shape[0]
    kw = W.shape[1]
    _ = W.shape[2]
    c_new = W.shape[3]

    # retrieve strides
    sh = stride[0]
    sw = stride[1]

    """
    ph: pad height
    pw: pad width
    Calculate padding for height (pad_height) and width (pad_width)
    The padding is calculated such that the output dimensions
        ...would be the same as input dimensions
    use np.ceil to round up to nearest whole number
        (can't have fractional padding)
    """
    # Compute padding dimensions
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:  # padding == "valid"
        # In case of 'valid' padding, no additional padding is added
        ph = pw = 0

    """
    output_height and output_width explanation:
    Compute the dimensions of the output volume from the convolution operation
    The output height and width are calculated based on the input dimensions,
        filter size, padding and stride
    Integer division (//) is used to ensure output dimensions are whole numbers

    OUTPUT HEIGHT FUNCTION
    h_prev + 2 * ph - kh:
        calculates effective height of input after padding (2 * ph) is added
        ...from this, the height of the filter (kh) is subtracted
        ...this gives the number of valid positions the filter
            ...can be placed on the input along the height dimension
    '// sh + 1': integer division by the stride (sh) along the height + 1
        calculates how many steps the filter can take along height of the input
        ...(including the initial position)
        ...each step of the filter results in one output feature
        ...this gives the height of the output feature map
    OUTPUT WIDTH FUNCTION:
    the same logic for output_height is used in calculating output_width
    """
    output_height = (h_prev + 2 * ph - kh) // sh + 1
    output_width = (w_prev + 2 * pw - kw) // sw + 1

    # Initialize the output volume Z with zeros
    Z = np.zeros((m, output_height, output_width, c_new))

    # Create A_prev_pad by padding A_prev
    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    # loop over the vertical (h), then horizontal (w), then over channels (c)
    for i in range(output_height):
        for j in range(output_width):
            for k in range(c_new):
                # Find the corners of the current slice
                start_h = i * sh
                start_w = j * sw
                end_h = start_h + kh
                end_w = start_w + kw

                # Use the corners to define the slice from A_prev_pad
                a_slice_prev = A_prev_pad[:, start_h:end_h, start_w:end_w, :]

                # Convolve the (3D) slice with the correct filter W and bias b,
                # to get back one output neuron
                weights = W[:, :, :, k]
                biases = b[0, 0, 0, k]
                Z[:, i, j, k] = np.sum(
                    a_slice_prev * weights, axis=(1, 2, 3)) + biases

    # Apply the activation function
    A = activation(Z)
    return A
