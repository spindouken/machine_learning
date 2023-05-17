#!/usr/bin/env python3
"""
calculates the derivative of a polynomial
"""


def poly_derivative(poly):
    """
    args: poly is a list of coefficients representing a polynomial

    the index of the list represents the power of x that
    ...the coefficient belongs to
    If poly is not valid, return None
    If the derivative is 0, return [0]

    Return: a new list of coefficients representing
    ...the derivative of the polynomial
    """
    if type(poly) is not list or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    return [poly[i] * i for i in range(1, len(poly))]
