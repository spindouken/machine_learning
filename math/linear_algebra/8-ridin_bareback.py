#!/usr/bin/env python3
"""multiply two 2D matrices"""


def mat_mul(mat1, mat2):
    """
    take two 2D matrices containing integers/floats and performs
    matrix multiplication
    If the matrices cannot be multiplied, the function returns None
    """
    # check if the matrices can be multiplied
    # if the number of columns in mat1 is not equal to the number
    # of rows in mat2, return None
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize the resulting matrix with zeros
    result = [[0 for column in range(len(mat2[0]))] for
              column in range(len(mat1))]

    # matrix multiplication
    for rowIndex in range(len(mat1)):
        for columnIndex in range(len(mat2[0])):
            for elementIndex in range(len(mat1[0])):
                result[rowIndex][columnIndex] += (
                    mat1[rowIndex][elementIndex] *
                    mat2[elementIndex][columnIndex]
                )
    return result
