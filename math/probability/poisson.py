#!/usr/bin/env python3
"""create class Poisson that represents a poisson distribution"""


class Poisson:
    """represents a poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """
        initialize Poisson class
        data will come in the form of a list
        lambtha will be of float value
        lambtha: the expected number of occurences in a given time
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        else:
            e = 2.7182818285
            factorial_k = 1
            for i in range(1, k + 1):
                factorial_k *= i
            pmf = (self.lambtha**k * e**-self.lambtha) / factorial_k
            return pmf

    def cdf(self, k):
        """calculates the value of the CDF for a given number of successes"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        else:
            cdf = 0
            for i in range(k + 1):
                cdf += self.pmf(i)
            return cdf
