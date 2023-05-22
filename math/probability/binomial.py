#!/usr/bin/env python3
"""
represents a binomial distribution
"""


class Binomial:
    """represents a binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """
        data: a list of the data to be used to estimate the distribution
        n: the number of Bernoulli trials
        p: the probability of a “success”
        Utilizes method of moments to estimate n and p
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

            # METHOD OF MOMENTS UTILIZED BELOW
            # calculate sample mean
            sample_mean = sum(data)/len(data)

            # calculate the sample variance
            sample_variance = 0
            for val in data:
                sample_variance = sample_variance + (val - sample_mean) ** 2
            sample_variance = sample_variance / len(data)

            # estimate q (1-p) using relationship between mean and variance
            q_estimate = sample_variance / sample_mean

            # estimate p using the estimate of q
            p_estimate = 1 - q_estimate

            # estimate n using n, p, and the mean for a binomial distribution
            n_estimate = (sum(data) / p_estimate) / len(data)

            # assign estimates
            self.n = int(round(n_estimate))
            self.p = float(sample_mean/self.n)
