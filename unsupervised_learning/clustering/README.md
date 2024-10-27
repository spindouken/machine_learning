### Clustering (Implemented from Scratch with NumPy and Using Scikit-learn)

#### Project Summary

The project focuses on implementing clustering algorithms, specifically K-means and Gaussian Mixture Models (GMM), from scratch using Python and NumPy, as well as utilizing Scikit-learn for enhanced functionality. It covers initialization, calculation of variance, optimization of cluster numbers, and evaluation of model performance using metrics like BIC (Bayesian Information Criterion).

---

#### Task Summaries

0. **Initialize K-means**
    - Initializes cluster centroids for K-means using a multivariate uniform distribution based on the dataset's minimum and maximum values. Utilizes NumPy's `random.uniform`.

1. **K-means**
    - Implements the K-means clustering algorithm, initializing centroids, updating them based on data points, and handling empty clusters. Uses NumPy and requires at most 2 loops.

2. **Variance**
    - Calculates the total intra-cluster variance of a dataset given the data points and centroids without using loops. Returns the variance value.

3. **Optimize k**
    - Analyzes different cluster sizes to determine the optimal number of clusters by calculating the variance for each size. Utilizes the K-means and variance functions with at most 2 loops.

4. **Initialize GMM**
    - Initializes variables for a Gaussian Mixture Model (GMM), including priors, centroid means (using K-means), and covariance matrices. Returns initialized parameters without loops.

5. **PDF**
    - Calculates the probability density function of a Gaussian distribution for given data points, mean, and covariance without using loops. Ensures all values are above a specified minimum.

6. **Expectation**
    - Performs the expectation step in the EM algorithm for GMM, calculating posterior probabilities and the total log likelihood, allowing for at most 1 loop.

7. **Maximization**
    - Executes the maximization step in the EM algorithm for GMM, updating priors, centroid means, and covariance matrices based on posterior probabilities, using at most 1 loop.

8. **EM**
    - Implements the Expectation-Maximization algorithm for GMM, iterating until convergence or a specified number of iterations. Prints log likelihood at intervals if verbose.

9. **BIC**
    - Finds the best number of clusters for GMM using Bayesian Information Criterion across specified cluster sizes. Computes log likelihood and BIC values, allowing for at most 1 loop.

10. **Hello, sklearn!**
    - Re-implements the K-means algorithm using Scikit-learn's clustering module. Returns the centroids and cluster assignments.

11. **GMM**
    - Calculates a GMM from a dataset using Scikit-learn's mixture module, returning priors, centroid means, covariance matrices, cluster assignments, and BIC values.

12. **Agglomerative Clustering**
    - Implements an agglomerative clustering algorithm using Ward linkage. The primary goal was to cluster a dataset and visualize the results using a dendrogram.
        - **Function Definition**: The function `agglomerative(X, dist)` was created to perform the clustering.
        - **Inputs**:
            - `X`: A `numpy.ndarray` of shape (n, d) representing the dataset.
            - `dist`: The maximum cophenetic distance for all clusters.
        - **Output**: Returns `clss`, a `numpy.ndarray` of shape (n,) containing the cluster indices for each data point.
        - **Clustering Method**: The function utilizes Ward linkage to minimize the variance within clusters, which is a suitable method for hierarchical clustering.
        - **Dendrogram Visualization**: The function also displays a dendrogram where each cluster is represented in a different color, aiding in the visualization of the clustering structure.
