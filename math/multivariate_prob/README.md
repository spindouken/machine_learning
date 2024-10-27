# Multivariate Probability From Scratch 

## Project Summary

This project focuses on coding statistical operations related to multivariate normal distributions without using any libraries (except NumPy). Key tasks include the calculation of mean, covariance, and correlation matrices. It also involves the implementation of a class that represents a multivariate normal distribution, featuring methods for calculating the probability density function (PDF).

### Task Summaries

0. **Mean and Covariance**: 
   - Implements a function to calculate the mean and covariance of a dataset stored in a 2D NumPy array. It validates the input to ensure it is a 2D array with multiple data points and returns the mean and covariance matrices.

1. **Correlation**: 
   - Creates a function to compute the correlation matrix from a given covariance matrix. It includes input validation to ensure that the covariance matrix is a valid 2D square array.

2. **Initialize (multinormal.py)**: 
   - Defines a `MultiNormal` class that represents a multivariate normal distribution. The class constructor initializes the mean and covariance from a provided 2D NumPy array, ensuring proper input format and dimensions.

3. **PDF (multinormal.py)**: 
   - Updates the `MultiNormal` class with a method to calculate the probability density function (PDF) at a given data point. It validates the input to ensure it is a NumPy array of the correct shape and returns the computed PDF value.
