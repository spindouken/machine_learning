### Bayesian Probability From Scratch

#### Project Summary

This project focuses on Bayesian statistics, specifically calculating probabilities related to a clinical trial for a cancer drug. **All implementations are done from scratch using only NumPy**, with the goal of determining the likelihood, intersection, marginal probability, and posterior probability of patients developing severe side effects based on trial data.

#### Task Summaries

0. **Likelihood**: Implements a function to calculate the likelihood of observing a certain number of patients developing severe side effects given a total number of patients and a set of hypothetical probabilities. The function validates inputs and computes the likelihood based on a binomial distribution.

1. **Intersection**: Extends the previous function to calculate the intersection of obtaining specific data with various hypothetical probabilities and prior beliefs about those probabilities. It includes extensive input validation and raises appropriate errors if conditions are not met.

2. **Marginal Probability**: Calculates the marginal probability of obtaining data based on the number of patients who developed severe side effects and their prior beliefs. The function validates inputs and raises errors for incorrect formats or values.

3. **Posterior**: Calculates the posterior probability of developing severe side effects given the observed data and prior beliefs. It builds on the previous tasks by combining the likelihood and prior probabilities while ensuring robust input validation and error handling.
