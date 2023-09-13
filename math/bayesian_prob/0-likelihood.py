#!/usr/bin/env python3
"""
You are conducting a study on a revolutionary cancer drug and are looking
    to find the probability that a patient who takes this drug
    will develop severe side effects.
During your trials, n patients take the drug
    and x patients develop severe side effects.
You can assume that x follows a binomial distribution.
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data given various
        hypothetical probabilities of developing severe side effects
    
    x: number of patients that develop severe side effects
    n: total number of patients observed
    P: 1D numpy.ndarray containing the various hypothetical probabilities
        of developing severe side effects
    If n is not a positive integer, raise a ValueError with the message:
        'n must be a positive integer'
    If x is not an integer that is greater than or equal to 0,
        raise a ValueError with the message:
        'x must be an integer that is greater than or equal to 0'
    If x is greater than n, raise a ValueError with the message:
        'x cannot be greater than n'
    If P is not a 1D numpy.ndarray, raise a TypeError with the message:
        'P must be a 1D numpy.ndarray'
    If any value in P is not in the range [0, 1], raise a ValueError with the
        message:
        'All values in P must be in the range [0, 1]'
    Returns: a 1D numpy.ndarray containing the likelihood of obtaining the
        data, x and n, for each probability in P, respectively
    """
