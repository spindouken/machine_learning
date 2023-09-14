#!/usr/bin/env python3
"""
represents a Multivariate Normal distribution
"""
import numpy as np


def mean_cov(X):
    """
    X is a numpy.ndarray of shape (n, d)
        containing the data set:
            n: the number of data points
            d: the number of dimensions in each data point

    If X is not a 2D numpy.ndarray,
        raise a TypeError with the message:
        'X must be a 2D numpy.ndarray'
    If n is less than 2,
        raise a ValueError with the message:
            'X must contain multiple data points'

    Returns: mean, cov:
        mean is a numpy.ndarray of shape (1, d)
            containing the mean of the data set
        cov is a numpy.ndarray of shape (d, d)
            containing the covariance matrix of the data set

    formula for calculating the covariance matrix (cov):

    cov = (Xcentered.T @ Xcentered) / (n - 1)

    formula explained:
        Xcentered: data matrix after centering it
            around zero by subtracting the mean
            from each data point. Shape = (n, d)
        Xcentered.T: Transpose of Xcentered, changing its shape to (d, n)
        n: number of data points
        d: number of dimensions in each data point
        @: matrix multiplication
        (n - 1): used for the unbiased estimator
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("data must be a 2D numpy.ndarray")

    # Get the number of data points (n) and dimensions (d)
    n, d = X.shape

    if n < 2:
        raise ValueError("data must contain multiple data points")

    # calculate mean for each feature
    #   sum along each feature and divide by total number of data points
    mean = np.sum(X, axis=0) / n

    # center data by subtracting the mean from each feature
    Xcentered = X - mean

    # calculate the covariance matrix using matrix multiplication
    # -----------------------------------------------------------
    # multiplication of Xcentered.T shape (d, n) and Xcentered shape (n, d)
    #   gives shape of (d, d), the shape of the covariance matrix
    # divide by (n - 1) for the unbiased estimator
    covarianceMatrix = (Xcentered.T @ Xcentered) / (n - 1)

    return mean, covarianceMatrix


class MultiNormal:
    """
    represents a Multivariate Normal distribution
    """

    def __init__(self, data):
        """
        data is a numpy.ndarray of shape (d, n) containing the data set:
            n: the number of data points
            d: the number of dimensions in each data point
        If data is not a 2D numpy.ndarray, raise a TypeError with the message:
            'data must be a 2D numpy.ndarray'
        If n is less than 2, raise a ValueError with the message:
            'data must contain multiple data points'
        Sets public instance variables:
            mean - a numpy.ndarray of shape (d, 1) containing the mean of data
            cov - a numpy.ndarray of shape (d, d)
                containing the covariance matrix data
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # transpose data from (d, n) to (n, d) for mean_cov expected input
        mean, covariance = mean_cov(data.T)

        self.mean = mean.reshape(-1, 1)
        self.cov = covariance

    def pdf(self, x):
        """
        calculates the PDF at a data point

        x is a numpy.ndarray of shape (d, 1) containing the data point
            whose PDF should be calculated
            d: the number of dimensions of the Multinomial instance
        If x is not a numpy.ndarray, raise a TypeError with the message:
            'x must be a numpy.ndarray'
        If x is not of shape (d, 1), raise a ValueError with the message:
            'x must have the shape ({d}, 1)'

        Returns the value of the PDF
        formula for Multivariate Normal distribution PDF:
        f(x) = (1 / (sqrt((2 * pi) ** d * det(cov)))) *
                    exp((-1 / 2) * (x - u).T @ cov^-1 @ (x - u))
        d: number of dimensions in the Multinomial instance
        x: data
        u: mean
        cov: covariance matrix
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.cov.shape[0]

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        mean = self.mean
        covariance = self.cov

        PDF = (
            1 / (np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance)))
        ) * np.exp(-0.5 * ((x - mean).T @ np.linalg.inv(covariance) @ (x - mean)))
        return PDF
