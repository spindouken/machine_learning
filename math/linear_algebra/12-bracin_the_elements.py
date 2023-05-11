#!/usr/bin/env python3
import numpy as np
"""numpy does it all for us :)
All element-wise operations of matrices"""


def np_elementwise(mat1, mat2):
    """
    Numpy approach is best approach.
    Use numpy to perform all element-wise operations on two input matrices
    Mathematicians everywhere groan from the lack of calculation drudgery

    Solutions will be provided in a tuple
    """
    addition = np.add(mat1, mat2)

    subtraction = np.subtract(mat1, mat2)

    multiplication = np.multiply(mat1, mat2)

    division = np.divide(mat1, mat2)

    return addition, subtraction, multiplication, division
