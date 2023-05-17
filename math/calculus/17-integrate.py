#!/usr/bin/env python3
"""calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """
    args: poly is a list of coefficients representing a polynomial
    C is an integer representing the integration constant

    the index of the list represents the power of
    ...x that the coefficient belongs to
    If a coefficient is a whole number, it should be
    ...represented as an integer

    return: a new list of coefficients representing the
    ...integral of the polynomial
    If poly or C are not valid, return None
    """
    if type(poly) is not list or len(poly) == 0 or type(C) is not int:
        return None

    integral_coefficients = [C]

    for i in range(len(poly)):
        if type(poly[i]) not in [int, float]:
            return None
        integral_coefficients.append(poly[i] / (i + 1))

    for i in range(len(integral_coefficients)):
        if integral_coefficients[i] == int(integral_coefficients[i]):
            integral_coefficients[i] = int(integral_coefficients[i])

    return integral_coefficients
