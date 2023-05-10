#!/usr/bin/env python3
"""Add two 2D matrices and return the product"""


def add_matrices2D(mat1, mat2):
    """
    This function takes two input 2D matrices and adds them element-wise
    The matrices must have the same number of rows and columns,
    or return None. (element-wise only works for matrices of same
    number of rows and columns)"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[mat1[rowIndex][columnIndex] + mat2[rowIndex][columnIndex] for columnIndex in range(len(mat1[0]))] for rowIndex in range(len(mat1))]
