### Hyperparameter Tuning (Implemented from Scratch, Last Task Uses GPyOpt)

[Medium Article: Optimizing Alzheimer's Disease Classification Using Bayesian Optimization and Transfer Learning](https://medium.com/@masonthecount/optimizing-alzheimers-disease-classification-using-bayesian-optimization-and-transfer-learning-3f9ed8cbad56)

#### Project Summary

This project involves implementing a Gaussian Process for optimization tasks using Bayesian Optimization techniques. Key concepts covered include:

1. **Gaussian Process Initialization**: Creating a class to represent a 1D Gaussian process with a covariance kernel.
2. **Prediction with Gaussian Process**: Adding functionality to predict mean and variance for new sample points.
3. **Updating Gaussian Process**: Implementing a method to incorporate new samples into the Gaussian process.
4. **Bayesian Optimization Initialization**: Setting up a class for Bayesian optimization with specified bounds and initial samples.
5. **Acquisition Function**: Implementing an expected improvement acquisition function to determine the next sampling point.
6. **Optimization Process**: Developing a method to iteratively optimize the black-box function, stopping when already sampled points are proposed.

---

#### Task Summaries

0. **Initialize Gaussian Process**
    - Create a `GaussianProcess` class that initializes with given inputs and calculates a covariance kernel matrix using the Radial Basis Function (RBF) kernel.

1. **Gaussian Process Prediction**
    - Update the `GaussianProcess` class to include a `predict` method that estimates the mean and variance for new sample points.

2. **Update Gaussian Process**
    - Add an `update` method to the `GaussianProcess` class, allowing it to incorporate a new sample point and update its internal attributes.

3. **Initialize Bayesian Optimization**
    - Create a `BayesianOptimization` class that sets up the optimization process using a Gaussian process. It initializes with a black-box function, initial samples, and other parameters.

4. **Bayesian Optimization - Acquisition**
    - Enhance the `BayesianOptimization` class by adding an `acquisition` method to compute the next sample point using the Expected Improvement acquisition function.

5. **Bayesian Optimization**
    - Implement an `optimize` method within the `BayesianOptimization` class to perform the optimization process over a specified number of iterations, halting if a sampled point is repeated. This task uses GPyOpt for the optimization process.
