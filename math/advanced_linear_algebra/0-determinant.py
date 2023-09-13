#!/usr/bin/env python3
"""
Calculate the determinant of a matrix using simple operations
"""


def determinant(matrix):
    """
    calculates the determinant of a matrix

    matrix is a list of lists whose determinant should be calculated
    If matrix is not a list of lists,
        raise a TypeError with the message:
        'matrix must be a list of lists'
    If matrix is not square,
        raise a ValueError with the message:
        'matrix must be a square matrix'
    The list [[]] represents a 0x0 matrix

    Returns: the determinant of matrix
    """
    # Check if the matrix is a list of lists
    if (
        not isinstance(matrix, list)
        or not all(isinstance(row, list) for row in matrix)
        or len(matrix) == 0
    ):
        raise TypeError("matrix must be a list of lists")

    # Special case: 0x0 matrix
    # In linear algebra, the determinant of a 1x1 matrix A = [a] is simply a.
    # For a 0x0 matrix, the matrix is essentially an "empty product," which
    # is defined to have a value of 1 by mathematical convention. This is
    # consistent with recursive expansion methods like Laplace's expansion.
    # Therefore, the determinant of an empty 0x0 matrix is defined to be 1.
    if matrix == [[]]:
        return 1

    # Check if the matrix is square
    rows = len(matrix)
    if not all(len(row) == rows for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base cases for 1x1 and 2x2 matrices
    #     determinant can be calculated w/o need for any loop operations
    #     (saves time and improves accuracy)
    if rows == 1:
        return matrix[0][0]
    if rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Rule of Sarrus for 3x3 matrices
    if rows == 3:
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        g, h, i = matrix[2]
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    # Recursive expansion using Laplace's formula
    detValue = 0
    for elementIndex in range(rows):
        # calculate minor matrix by removing the current row and column
        minorMatrix = [
            row[:elementIndex] + row[elementIndex + 1 :] for row in matrix[1:]
        ]

        # calculate cofactor for the current element in the matrix
        cofactor = ((-1) ** elementIndex) * matrix[0][elementIndex]

        # recursive call to find the determinant of the minor matrix
        detValue += cofactor * determinant(minorMatrix)

    return detValue
