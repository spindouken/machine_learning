#!/usr/bin/env python3
"""
"""


class Binomial:
    """placeholder"""
    def __init__(self, data=None, n=1, p=0.5):
        """
        data: a list of the data to be used to estimate the distribution
        n: the number of Bernoulli trials
        p: the probability of a “success”
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.n = int(n)
                self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            trials = len(data)
            successes = self.p * trials
            self.p = successes / trials
            self.n = round(self.p * trials)
            self.p = self.n / trials
