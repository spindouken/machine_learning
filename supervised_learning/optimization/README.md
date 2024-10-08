# Machine Learning Optimization Techniques

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)

## Project Overview

This project involved implementing various concepts related to data normalization, shuffling, mini-batching, and optimization algorithms in machine learning using TensorFlow and NumPy. It focuses on functions that calculate normalization constants, shuffle data, create mini-batches, and apply optimization techniques like momentum, RMSProp, and Adam. Additionally, it addresses batch normalization within neural networks.

## Key Features

- **Normalization Constants**: Calculates the mean and standard deviation for each feature in a given dataset to standardize inputs, improving model training stability.
- **Data Normalization**: Normalizes datasets based on the calculated means and standard deviations, allowing for standardized inputs that enhance learning.
- **Data Shuffling**: Implements a method to shuffle data matrices identically to maintain correspondence between features and labels.
- **Mini-Batching**: Creates mini-batches for training, allowing for efficient processing of data in smaller chunks while maintaining shuffling.
- **Optimization Algorithms**: Implements various optimization techniques including momentum, RMSProp, and Adam, each aimed at improving convergence during training.
- **Batch Normalization**: Establishes a batch normalization layer in TensorFlow, which incorporates trainable parameters to normalize outputs within neural networks.

## Prerequisites

To run this project, I installed:

- Python 3.x
- TensorFlow
- NumPy

## Task Summaries

0. **Normalization Constants**:
   Calculated the mean and standard deviation for each feature in a given matrix using NumPy.

1. **Normalize**:
   Normalized a matrix based on provided means and standard deviations, returning the standardized matrix.

2. **Shuffle Data**:
   Shuffled two matrices identically to maintain their correspondence, using NumPy's random permutation.

3. **Mini-Batch**:
   Created mini-batches from input data and labels for training, allowing for a smaller final batch, utilizing the shuffle data function.

4. **Moving Average**:
   Computed a weighted moving average of a dataset with bias correction.

5. **Momentum**:
   Updated a variable using gradient descent with momentum optimization.

6. **Momentum Upgraded**:
   Set up a TensorFlow operation for gradient descent with momentum.

7. **RMSProp**:
   Updated a variable using the RMSProp optimization algorithm.

8. **RMSProp Upgraded**:
   Configured a TensorFlow operation for the RMSProp optimization.

9. **Adam**:
   Updated a variable using the Adam optimization algorithm.

10. **Adam Upgraded**:
   Established a TensorFlow operation for the Adam optimization.

11. **Batch Normalization Upgraded**:
   Created a batch normalization layer in TensorFlow, incorporating trainable parameters and an activation function.
