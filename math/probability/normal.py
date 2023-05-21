#!/usr/bin/env python3
"""create class Normal that represents a normal distribution
A normal distribution is a type of statistical distribution where data
tends to cluster around a mean or average value.
"""


class Normal:
    """represents a normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        initialize Normal distribution class with data, mean, and stddev

        data: list of data points from which to estimate the distribution
        ...if no data is provided, the distribution will initialized with
        ...default provided values

        stddev: standard deviation of the distribution (measure of variation)
        ...if no data is provided, use default provided stddev value
        """
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
            # calculate squared differences from the mean
            deviation = [(x - self.mean) ** 2 for x in data]
            # calculate variance (average of the squared differences)
            variance = (sum(deviation) / len(data))
            # calculate stddev (sqrt of the variance)
            self.stddev = variance ** 0.5
