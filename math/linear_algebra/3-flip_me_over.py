#!/usr/bin/env python3
"""Transpose a given matrix"""


def matrix_transpose(matrix):
    """
    Takes a 2D matrix as input and returns the transposed matrix.
    This is done by swapping the rows and colums.
    """
    transposedMatrix = [[matrix[columnIndex][rowIndex]
                        for columnIndex in range(len(matrix))]
                        for rowIndex in range(len(matrix[0]))]
    return transposedMatrix
