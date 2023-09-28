#!/usr/bin/env python3

import numpy as np
expectation = __import__('6-expectation').expectation

if __name__ == "__main__":
    X = np.random.randn(100, 6)
    pi = np.random.randn(5)
    pi = pi / np.sum(pi)
    S = np.random.randn(5, 6, 6)
    print(expectation(X, pi, 'hello', S))
    print(expectation(X, pi, np.array([1, 2, 3, 4, 5]), S))
    print(expectation(X, pi, np.array([[[1, 2, 3, 4, 5]]]), S))
    print(expectation(X, pi, np.random.randn(5, 5), S))
