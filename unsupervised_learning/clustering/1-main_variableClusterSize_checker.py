#!/usr/bin/env python3

import numpy as np
kmeans = __import__('1-kmeans').kmeans

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=150)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=150)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    C, clss = kmeans(X, 4, iterations=5)
    print(C)
    print(clss)
