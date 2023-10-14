#!/usr/bin/env python3

import numpy as np
GP = __import__('0-gp').GaussianProcess

def f(x):
    """our 'black box' function"""
    return np.cos(5*x) + 2*np.cos(-2*x)

np.random.seed(5)
X_init = np.random.uniform(-np.pi, 2*np.pi, (5, 1))
Y_init = f(X_init)
gp = GP(X_init, Y_init, l=0.5, sigma_f=1.5)
print(gp.X is X_init)
print(gp.Y is Y_init)
print(gp.l)
print(gp.sigma_f)
