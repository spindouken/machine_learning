#!/usr/bin/env python3

import numpy as np
maximization = __import__('7-maximization').maximization

if __name__ == "__main__":
    X = np.random.randn(100, 6)
    print(maximization(X, 'hello'))
    print(maximization(X, np.array([1, 2, 3, 4, 5])))
    print(maximization(X, np.array([[[1, 2, 3, 4, 5]]])))
    g = np.random.randn(5, 90)
    g = g / np.sum(g, axis=0, keepdims=True)
    print(maximization(X, g))
    print(maximization(X, np.random.randn(5, 100)))
