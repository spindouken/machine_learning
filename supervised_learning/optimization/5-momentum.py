#!/usr/bin/env python3
"""Write the function that updates a variable using
the gradient descent with momentum optimization algorithm"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
