#!/usr/bin/env python3
"""
calculates the marginal probability of obtaining the data
"""
import numpy as np


def marginal(x, n, P, Pr):
    """
    calculates the marginal probability of obtaining the data

    x: number of patients that develop severe side effects
    n: total number of patients observed
    P: 1D numpy.ndarray containing the various hypothetical probabilities
        of developing severe side effects
    Pr: 1D numpy.ndarray containing the prior beliefs of P
    If n is not a positive integer, raise a ValueError with the message:
        'n must be a positive integer'
    If x is not an integer that is greater than or equal to 0,
        raise a ValueError with the message:
        'x must be an integer that is greater than or equal to 0'
    If x is greater than n, raise a ValueError with the message:
        'x cannot be greater than n'
    If P is not a 1D numpy.ndarray, raise a TypeError with the message:
        'P must be a 1D numpy.ndarray'
    If Pr is not a numpy.ndarray with the same shape as P, raise a TypeError
        with the message:
        'Pr must be a numpy.ndarray with the same shape as P'
    If any value in P or Pr is not in the range [0, 1], raise a ValueError
        with the message:
        'All values in {P} must be in the range [0, 1] where {P} is the
            incorrect variable
    If Pr does not sum to 1, raise a ValueError with the message:
        'Pr must sum to 1'
    All exceptions should be raised in the above order

    Returns: the marginal probability of obtaining x and n
        P is the prior probability of each probability in P
        Pr is the prior probability of each probability in Pr
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0'
        )
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError('All values in P must be in the range [0, 1]')
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(Pr.sum(), 1):
        raise ValueError('Pr must sum to 1')
