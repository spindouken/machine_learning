#!/usr/bin/env python3
def matrix_transpose(matrix):
    return [[matrix[columnIndex][rowIndex] for columnIndex in range(len(matrix))] for rowIndex in range(len(matrix[0]))]
