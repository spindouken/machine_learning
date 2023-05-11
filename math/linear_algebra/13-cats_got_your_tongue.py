#!/usr/bin/env python3
"""concatenate two arrays along a specific axis using numpers"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    concatenate two numpy arrays along a specific axis
    """
    return np.concatenate((mat1, mat2), axis)
