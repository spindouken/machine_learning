#!/usr/bin/env python3
"""
calculates the sum of squares of integers from 1 to n,
...where n is a positive integer.
"""


def summation_i_squared(n):
    """
    args: n as integer is the stopping condition

    Return:
    sum of squares of the first n natural numbers: (n*(n+1)*(2*n+1))//6
    None if n is not a valid number or not greater than 0.
    """
    if not isinstance(n, int) or n < 1:
        return None
    else:
        return (n*(n+1)*(2*n+1))//6
