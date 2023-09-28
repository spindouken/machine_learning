#!/usr/bin/env python3

import numpy as np
initialize = __import__('4-initialize').initialize

if __name__ == '__main__':
    np.random.seed(1)
    a = np.random.multivariate_normal([30, 40, 50], 70 * np.eye(3) + 5, size=10000)
    b = np.random.multivariate_normal([5, 25, 10], 5 * np.eye(3) + 10, size=750)
    c = np.random.multivariate_normal([60, 30, -20], 10 * np.eye(3), size=750)
    X = np.concatenate((a, b, c), axis=0)
    np.random.shuffle(X)
    pi, m, S = initialize(X, 3)
    print(pi)
    print(m)
    print(S)
