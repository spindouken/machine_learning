#!/usr/bin/env python3
"""
determines if gradient descent should stop early
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    cost: the current validation cost of the neural network
    opt_cost: the lowest recorded validation cost of the network
    threshold: the threshold used for early stopping
    patience: the patience count used for early stopping
    count: count of how long the threshold has not been met
    Returns a boolean of whether the network
        should be stopped early, followed by the updated count
    """
    costGap = opt_cost - cost

    if costGap > threshold:
        count = 0
    else:
        count += 1

    if count < patience:
        return(False, count)
    else:
        return(True, count)
