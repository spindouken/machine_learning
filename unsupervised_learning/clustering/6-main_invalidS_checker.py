#!/usr/bin/env python3

import numpy as np
expectation = __import__('6-expectation').expectation

if __name__ == "__main__":
    X = np.random.randn(100, 6)
    pi = np.random.randn(5)
    pi = pi / np.sum(pi)
    m = np.random.randn(5, 6)
    print(expectation(X, pi, m, 'hello'))
    print(expectation(X, pi, m, np.array([1, 2, 3, 4, 5])))
    print(expectation(X, pi, m, np.array([[1, 2, 3, 4, 5]])))
    print(expectation(X, pi, m, np.random.randn(4, 6, 6)))
    print(expectation(X, pi, m, np.random.randn(5, 5, 6)))
    print(expectation(X, pi, m, np.random.randn(5, 6, 5)))
