#!/usr/bin/env python3
"""
determines if a markov chain is absorbing
"""
import numpy as np


def absorbing(P):
    """
    P is a is a square 2D numpy.ndarray of shape (n, n) representing the
        transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure
    """
    n = P.shape[0]
    # np.diag(P) extracts the diagonal of P and enumerate gives
    #   the index of each element in the diagonal and the element itself
    # if the element is 1, it is an absorbing state
    #   and is added to list absorbingStates
    absorbingStates = [
        stateIndex for stateIndex, value in enumerate(np.diag(P)) if value == 1
    ]

    # if no absorbing states, return False
    if len(absorbingStates) == 0:
        return False

    return True
