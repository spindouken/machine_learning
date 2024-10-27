### Dimensionality Reduction from Scratch

#### Project Summary

This project implements Principal Component Analysis (PCA) from scratch, a technique used in unsupervised learning for dimensionality reduction. The goal is to transform a dataset into a lower-dimensional space while preserving as much variance as possible.

#### Task Summaries

0. **PCA**: Develops a function to perform PCA on a dataset and return a weights matrix that maintains a specified fraction of the variance. The function validates input data, ensuring the dataset has zero mean and computes the necessary eigenvalues and eigenvectors to determine the principal components.

1. **PCA v2**: Extends the initial PCA implementation to allow for direct specification of the new dimensionality for the transformed dataset. It returns the transformed dataset in the specified dimensionality, providing more flexibility for different analysis needs. This version also ensures that the inputs are properly formatted and computes the transformation using the principal components derived from the covariance matrix.

#### Key Implementation Details

- **Input Validation**: Both tasks validate inputs, checking for appropriate shapes and conditions such as mean zero.
- **Eigenvalue Decomposition**: Utilizes NumPy's linear algebra functions to compute eigenvalues and eigenvectors, crucial for PCA.
- **Dimensionality Reduction**: The methods focus on preserving variance and effectively reducing the number of features while maintaining the integrity of the data's structure.
