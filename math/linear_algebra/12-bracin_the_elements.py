#!/usr/bin/env python3
"""numpy does it all for us :)
All element-wise operations of matrices"""


def np_elementwise(mat1, mat2):
    """
    Numpy approach is best approach.
    Use numpy to perform all element-wise operations on two input matrices
    Mathematicians everywhere groan from the lack of calculation drudgery

    Solutions will be provided in a tuple
    """
    addition = mat1 + mat2

    subtraction = mat1 - mat2

    multiplication = mat1 * mat2

    division = mat1 / mat2

    return addition, subtraction, multiplication, division
