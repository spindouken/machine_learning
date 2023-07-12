#!/usr/bin/env python3
"""that performs back propagation over a pooling
layer of a neural network"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new)
        ...containing the partial derivatives with respect
        ...to the output of the pooling layer
        m: the number of examples
        h_new: the height of the output
        w_new: the width of the output
        c: the number of channels
    A_prev a numpy.ndarray of shape (m, h_prev, w_prev, c)
        ...containing the output of the previous layer
        h_prev: the height of the previous layer
        w_prev: the width of the previous layer
    kernel_shape: a tuple of (kh, kw)
        ...containing the size of the kernel for the pooling
        kh: the kernel height
        kw: the kernel width
    stride: a tuple of (sh, sw)
        ...containing the strides for the pooling
    sh: the stride for the height
    sw: the stride for the width
    mode: a string containing either max or avg,
        ...indicating whether to perform maximum or average pooling
    Returns: the partial derivatives w/ respect to the previous layer (dA_prev)
    """
    m = dA.shape[0]
    h_new = dA.shape[1]
    w_new = dA.shape[2]
    c_new = dA.shape[3]

    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c = A_prev.shape[3]

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    # Initialize dA_prev with zeros
    dA_prev = np.zeros_like(A_prev)

    # Loop over the training examples
    for batch_index in range(m):
        # Loop over the height (h), then width (w), then channels (c)
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    # Find the corners of the current slice
                    start_h = h * sh
                    start_w = w * sw
                    end_h = start_h + kh
                    end_w = start_w + kw

                    # Compute the backward propagation in both modes.
                    if mode == 'avg':
                        # Get the value a from dA for the current position
                        da = dA[batch_index, h, w, ch]

                        # Define the shape of the kernel as k
                        k = kernel_shape

                        # Distribute it to get the correct slice of dA_prev
                        # i.e. Add the distributed value of da
                        average_da = da / (kh * kw)
                        a_slice = dA_prev[
                            batch_index, start_h:end_h, start_w:end_w, ch]
                        # Add the distributed value to get the new slice
                        dA_prev[
                            batch_index, start_h:end_h, start_w:end_w, ch
                            ] += np.ones(k) * average_da

                    elif mode == 'max':
                        # Select the slice from the previous activation layer
                        # that was used to compute the pooling operation
                        a_prev_slice = A_prev[
                            batch_index, start_h:end_h, start_w:end_w, ch]
                        # Create a mask to distribute the gradient
                        #   ...mask should be the same size as a_prev_slice and
                        # every entry of a_prev_slice that was not the max
                        #   ...should be set to 0 in mask, all others set to 1
                        mask = a_prev_slice == np.max(a_prev_slice)

                        # Distribute it to get the correct slice of dA_prev
                        # i.e. Add mask multiplied by correct entry of dA
                        dA_prev[
                            batch_index, start_h:end_h, start_w:end_w, ch
                            ] += mask * dA[batch_index, h, w, ch]

    return dA_prev
