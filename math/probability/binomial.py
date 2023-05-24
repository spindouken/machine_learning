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

    def factorial(self, k):
        """calculates the factorial of a number"""
        if k == 0:
            return 1
        else:
            return k * self.factorial(k - 1)

    def combination(self, n, k):
        """calculates the combination of n and k"""
        return self.factorial(n) / (self.factorial(k) * self.factorial(n - k))

    def pmf(self, k):
        """calculates the value of the PMF for a given number of successes"""
        k = int(k)

        if k < 0 or k > self.n:
            return 0

        # calculate binomial coefficient (n choose k)
        binomialCoef = self.combination(self.n, k)

        pmf = binomialCoef * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return pmf

    def cdf(self, k):
        """calculates the value of the CDF for a given number of successes
        k: the number of successes
        returns the CDF (Cumulative Distribution Function) for int k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)

        return cdf
