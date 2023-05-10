#!/usr/bin/env python3
"""Concatenate two 2D matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Take two 2D matrices containing integers/floats and concatenate
    them along the specified axis.
    If the matrices cannot be concatenated, the function returns None.
    """
    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    if axis == 1 and len(mat1) != len(mat2):
        return None

    if axis == 0:
        # concatenate the matrices along rows (axis 0)
        # by creating a new matrix with the rows of mat1
        # followed by the rows of mat2
        concatenatedMatrix = mat1 + mat2
    else:
        # concatenate the matrices along columns (axis 1)
        # by creating a new matrix with the columns of mat1
        # followed by the columns of mat2 for each row
        concatenatedMatrix = [row1 + row2 for row1, row2 in zip(mat1, mat2)]

    return concatenatedMatrix
