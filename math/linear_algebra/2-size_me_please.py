#!/usr/bin/env python3
def matrix_shape(matrix):
    """Returns the dimensions of a given matrix"""
    matrixDimensions = []
    while isinstance(matrix, list):
        matrixDimensions.append(len(matrix))
        matrix = matrix[0]
    return matrixDimensions
