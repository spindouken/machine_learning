#!/usr/bin/env python3
"""create a class Exponential that represents an exponential distribution"""


class Exponential:
    """represents an exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            lambthat = lambtha
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))
