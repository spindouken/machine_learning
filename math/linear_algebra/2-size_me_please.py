#!/usr/bin/env python3
"""Find the shape of a matrix and
return the dimensions as a list of integers"""


def matrix_shape(matrix):
    """Returns the dimensions of a given matrix"""
    matrixDimensions = []
    while isinstance(matrix, list):
        matrixDimensions.append(len(matrix))
        matrix = matrix[0]
    return matrixDimensions
