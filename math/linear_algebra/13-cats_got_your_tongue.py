#!/usr/bin/env python3
import numpy as np
"""concatenate two arrays along a specific axis using numpers"""


def np_cat(mat1, mat2, axis=0):
    """
    concatenate two numpy arrays along a specific axis
    """
    return np.concatenate((mat1, mat2), axis)
