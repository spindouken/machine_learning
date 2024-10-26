# Probability from Scratch with Python

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)

## Project Overview

This project covers the implementation of various probability distributions: Poisson, Exponential, Normal, and Binomial. Each distribution class includes methods for initialization, probability mass function (PMF), cumulative distribution function (CDF), and relevant calculations like z-scores and PDF.

## Key Features

- **Poisson Distribution**: Class for calculating the PMF and CDF, with validation for initialization parameters.
- **Exponential Distribution**: Class for calculating the PDF and CDF, including validation for the rate parameter.
- **Normal Distribution**: Class for calculating z-scores, x-values, PDF, and CDF, with validation for the mean and standard deviation.
- **Binomial Distribution**: Class for calculating the PMF and CDF, ensuring valid inputs for the number of trials and probability of success.

## Prerequisites

- Python 3.x

## Task Summaries

0. **Initialize Poisson**: 
    - Create a `Poisson` class to represent a Poisson distribution with an initializer that sets `lambtha` based on provided data or a specified value. Validates input for positive values and appropriate data types.

1. **Poisson PMF**: 
    - Add an instance method `pmf` to the `Poisson` class to calculate the PMF for a given number of successes, ensuring input is an integer and within valid range.

2. **Poisson CDF**: 
    - Extend the `Poisson` class with a `cdf` method to compute the CDF for a given number of successes, with similar input validation as PMF.

3. **Initialize Exponential**: 
    - Create an `Exponential` class with an initializer for `lambtha` based on data or a specified value, including input validation for data type and value.

4. **Exponential PDF**: 
    - Add a method `pdf` to the `Exponential` class to calculate the PDF for a given time period, ensuring proper range checks.

5. **Exponential CDF**: 
    - Extend the `Exponential` class with a `cdf` method to compute the CDF for a specified time period, with range validation.

6. **Initialize Normal**: 
    - Create a `Normal` class with an initializer that sets `mean` and `stddev` based on data or specified values, including validations for positive standard deviation.

7. **Normalize Normal**: 
    - Add methods `z_score` and `x_value` to the `Normal` class to calculate z-scores and x-values based on given inputs.

8. **Normal PDF**: 
    - Introduce a `pdf` method to the `Normal` class to calculate the PDF for a specified x-value.

9. **Normal CDF**: 
    - Extend the `Normal` class with a `cdf` method to compute the CDF for a given x-value.

10. **Initialize Binomial**: 
    - Create a `Binomial` class with an initializer for `n` and `p` based on data or specified values, including validations for positive n and valid probability.

11. **Binomial PMF**: 
    - Add a `pmf` method to the `Binomial` class to calculate the PMF for a specified number of successes, ensuring input validation.

12. **Binomial CDF**: 
    - Extend the `Binomial` class with a `cdf` method to compute the CDF for a given number of successes, utilizing the PMF method for calculations.
