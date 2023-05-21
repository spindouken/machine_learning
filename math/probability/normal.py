#!/usr/bin/env python3
"""create class normal that represents a normal distribution"""


class Normal:
    def __init__(self, data=None, mean=0., stddev=1.):
        """initialize Normal distribution class with data, mean, and stddev"""
        if data is None:
            self.mean = float(mean)
            self.stddev = float(stddev)
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            deviation = [(x - self.mean) ** 2 for x in data]
            variance = (sum(deviation) / len(data))
            self.stddev = variance ** 0.5            
