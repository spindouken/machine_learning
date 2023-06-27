#!/usr/bin/env python3
"""
 calculates the cost of a neural network with L2 regularization
"""


def l2_reg_cost(cost):
    """
    calculates the cost of a neural network with L2 regularization
    cost: tensor containing the cost of the network without L2 regularization
    Returns a tensor containing the cost
        of the network accounting for L2 regularization
    """
