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

    def z_score(self, x):
        """
        calculate the z-score of a given x-value

        z-score: measure of how many standard deviations x is from the mean

        z-score = subtract mean from x and divide the result by the stddev
        """
        z_score = (x - self.mean) / self.stddev
        return z_score

    def x_value(self, z):
        """
        calculate the x-value of a given z-score

        x-value: original value from a z-score
        x-value = multiply the z-score by the stddev and add the mean

        z: the z-score
        """
        x_value = z * self.stddev + self.mean
        return x_value

    def pdf(self, x):
        """
        calculate the pdf of a given x-value

        x: the x-value to find the probability density from
        e: base of the natural logoarithm (simplified Euler's number)
        mu: or μ stands for mean of the distribution
        sigma: or σ stands for stddev
        pdf: Probability Density Function (PDF)
        pdf formula for a normal distribution:
        (1 / sqrt(2 * pi * sigma^2)) * e^((-1/2) * ((x - mu) / sigma)^2)
        """
        pi = 3.1415926536
        e = 2.7182818285
        mu = self.mean
        sigma = self.stddev
        pdf = (1 / ((2 * pi * sigma ** 2) ** 0.5)) \
            * e ** ((-1/2) * ((x - mu) / sigma) ** 2)
        return pdf

    def cdf(self, x):
        """
        calculate the cdf of a given x-value

        x: the x-value to find the cumulative distribution from
        e: base of the natural logoarithm (simplified Euler's number)
        mu: or μ stands for mean of the distribution
        sigma: or σ stands for stddev
        cdf: Cumulative Distribution Function (CDF)
        """
        mu = self.mean
        sigma = self.stddev
        z_score = (x - mu) / (sigma * 2 ** 0.5)
        erf = (2 / 3.1415926536 ** 0.5) * (
            z_score -
            (z_score ** 3) / 3 +
            (z_score ** 5) / 10 -
            (z_score ** 7) / 42 +
            (z_score ** 9) / 216
        )
        cdf = 0.5 * (1 + erf)
        return cdf
