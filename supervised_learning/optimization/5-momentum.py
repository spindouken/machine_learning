#!/usr/bin/env python3
"""Write the function that updates a variable using
the gradient descent with momentum optimization algorithm"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    alpha: learning rate
    beta1: momentum weight
    var: numpy.ndarray containing the variable to be updated
    grad: numpy.ndarray containing the gradient of var
    v: previous first moment of var
    Returns the updated variable and the new moment, respectively
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
