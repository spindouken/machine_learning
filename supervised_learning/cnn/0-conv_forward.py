#!/usr/bin/env python3
"""performs forward propagation over
a convolutional layer of a neural network"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    perform forward propagation over a convolutional layer of a neural network

    Function variables explanation:
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
    Calculate padding for height (ph) and width (pw)
    padding is calculated such that the output dimensions would be the same
        as input dimensions

    ph: pad height
    pw: pad width

    SAME PADDING CALCULATION:
    'ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))' explanation:
    (h_prev - 1) * sh + kh:
        gives the total height that the input would have if
        we were to slide the filter across the entire height
        of the input with the given stride, sh.
        Subtracting h_prev from this value gives the total amount
        of extra space needed to be able to do this.
        Dividing by 2 ensures that this extra space is evenly distributed
        on both sides (top and bottom) of the input.
    np.ceil is used to round up to nearest whole number, in case the division
    results in a fractional number. This is because we can't have a fractional
    number of pixels for padding.
    The same logic applies to the calculation of 'pw', the width padding.

    VALID PADDING CALCULATION:
    In case of 'valid' padding, no additional padding is added.
    The filter does not go outside the bounds of the input,
        and every position at which the filter is
        applied is a 'valid' position within the input.
    """
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:  # padding == "valid" (no padding applied)
        ph = pw = 0

    """
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

    """
    Performs Convolution Operation / Filter Application / Kernel Loop
    Loop over the vertical (height), then horizontal (width),
        then over channels (c)
    This is done to apply the filter to every possible position in the input
    volume for each channel.

    'for h in range(output_height):'
    iterates over the height of the output volume. For each position,
    the filter is applied to the corresponding position in the input volume.

    'for w in range(output_width):'
    iterates over the width of the output volume.
    Combined with the outer loop,
        this allows us to cover every position in 2D space of the input volume

    'for c in range(c_new):':
    iterates over the channels of the output volume. For each channel,
    a different set of filters is applied.
    """
    # loop over the vertical (h), then horizontal (w), then over channels (c)
    for h in range(output_height):
        for w in range(output_width):
            for c in range(c_new):
                # calculate the starting position of the filter on the input
                #   for the current step of the convolution
                # starting position determined by current step of convolution
                # (h or w) multiplied by the stride (sh or sw)
                filter_start_height = h * sh
                filter_start_width = w * sw
                # calculate the ending position of the filter on the input
                #   for current step of the convolution
                # the ending position is determined by the starting pos
                #   + the size of the filter (kh or kw)
                filter_end_height = filter_start_height + kh
                filter_end_width = filter_start_width + kw

                # Extract a slice from the padded input volume.
                # The slice is the portion of the input volume that the filter
                #   will be applied to in the current step.
                # Use the corners to define the slice from A_prev_pad
                a_slice_prev = A_prev_pad[
                    :,
                    filter_start_height:filter_end_height,
                    filter_start_width:filter_end_width,
                    :]

                # Extract the filters and biases for the current channel.
                weights = W[:, :, :, c]
                biases = b[0, 0, 0, c]

                # Apply the filter to the slice of the input volume,
                #   sums up the resultsx and adds the bias.
                # The result is a single value that is the output of
                #   the convolution for the current position and channel.
                Z[:, h, w, c] = np.sum(
                    a_slice_prev * weights, axis=(1, 2, 3)) + biases

    # Apply the activation function
    A = activation(Z)
    return A
