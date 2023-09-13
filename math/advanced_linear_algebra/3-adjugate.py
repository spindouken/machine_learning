#!/usr/bin/env python3
"""
Calculate the determinant of a matrix using simple operations
"""


def validateMatrix(matrix):
    """
    validates if the input is a square matrix

    matrix: list of lists

    Raises:
        TypeError: If the matrix is not a list of lists.
        ValueError: If the matrix is not square or is empty.
    """
    if (
        not isinstance(matrix, list)
        or not all(isinstance(row, list) for row in matrix)
        or len(matrix) == 0
    ):
        raise TypeError("matrix must be a list of lists")

    rows = len(matrix)
    if not all(len(row) == rows for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")


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
    validateMatrix(matrix)

    if matrix == [[]]:
        return 1
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    rows = len(matrix)

    if rows == 1:
        return matrix[0][0]
    if rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    if rows == 3:
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        g, h, i = matrix[2]
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    # Recursive expansion using Laplace's formula
    detValue = 0
    for elementIndex in range(rows):
        # calculate minor matrix by removing the first row
        # with first row removed go through the column value of each row
        #     while checking if the column index is the same as the element
        #     index, if it is, skip it
        minorMatrix = [
            [
                columnValue
                for columnIndex, columnValue in enumerate(row)
                if columnIndex != elementIndex
            ]
            for row in matrix[1:]
        ]

        # calculate cofactor for the current element in the matrix
        cofactor = ((-1) ** elementIndex) * matrix[0][elementIndex]

        # recursive call to find the determinant of the minor matrix
        detValue += cofactor * determinant(minorMatrix)

    return detValue


def minor(matrix):
    """
    calculates the minor matrix of a matrix

    matrix is a list of lists whose minor matrix should be calculated
    If matrix is not a list of lists,
        raise a TypeError with the message:
            matrix must be a list of lists
    If matrix is not square or is empty,
        raise a ValueError with the message
            matrix must be a non-empty square matrix

    Returns: the minor matrix of matrix
    """
    validateMatrix(matrix)

    if matrix == [[]]:
        return 1
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]

    rows = len(matrix)

    minorMatrix = []
    # loop through each row in the matrix
    for i in range(rows):
        minorRow = []
        # loop through each column in the matrix
        for j in range(rows):
            # create sub-matrix by removing the current row and column
            subMatrix = []
            for x in range(rows):
                if x == i:  # skip to next iteration of x = i element in matrix
                    continue
                subRow = []  # create a new row for the sub-matrix
                for y in range(rows):  # loop through column values in row
                    if y == j:  # skip to next iteration if y = j ele in matrix
                        continue
                    subRow.append(matrix[x][y])  # add element to sub-row
                subMatrix.append(subRow)  # add sub-row to sub-matrix

            # Calculate the determinant of the sub-matrix
            minorValue = determinant(subMatrix)
            minorRow.append(minorValue)  # add minor value to minor row
        # add minor row to minor matrix
        minorMatrix.append(minorRow)

    return minorMatrix


def cofactor(matrix):
    """
    calculates the cofactor of a matrix

    matrix is a list of lists whose
        cofactor matrix should be calculated
    If matrix is not a list of lists,
        raise a TypeError with the message
            'matrix must be a list of lists'
    If matrix is not square or is empty,
        raise a ValueError with the message
        'matrix must be a non-empty square matrix'
    Returns: the cofactor matrix of matrix
    """
    validateMatrix(matrix)

    if matrix == [[]]:
        return 1
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]

    rows = len(matrix)

    cofactorMatrix = []
    # loop through each row in the matrix
    for i in range(rows):
        cofactorRow = []
        # loop through each column value in the matrix
        for j in range(rows):
            subMatrix = []
            # loop through each row to create the sub-matrix
            for x in range(rows):
                if x == i:  # skip the current row
                    continue
                subRow = []
                # loop through each column value to create the sub-row
                for y in range(rows):
                    if y == j:  # Skip the current column
                        continue
                    subRow.append(
                        matrix[x][y]
                    )  # Add the element to the sub-row
                subMatrix.append(subRow)  # Add the sub-row to the sub-matrix

            # Calculate the determinant of the sub-matrix
            minorValue = determinant(subMatrix)
            # Calculate the cofactor and add it to the cofactor row
            cofactorRow.append(minorValue * (-1) ** (i + j))
        # Add the cofactor row to the cofactor matrix
        cofactorMatrix.append(cofactorRow)
    return cofactorMatrix


def adjugate(matrix):
    """
    calculates the adjugate matrix of a matrix

    matrix is a list of lists
        whose adjugate matrix should be calculated
    If matrix is not a list of lists,
        raise a TypeError with the message:
            matrix must be a list of lists
    If matrix is not square or is empty,
        raise a ValueError with the message:
            matrix must be a non-empty square matrix
    Returns: the adjugate matrix of matrix
    """
