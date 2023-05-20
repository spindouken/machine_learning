#!/usr/bin/env python3
"""create class Poisson that represents a poisson distribution"""
import numpy as numpers


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
