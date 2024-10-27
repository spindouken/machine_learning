# Regularization

## Supplementary Medium Article
https://medium.com/@masonthecount/why-regularization-is-important-9aed9ecc25e1

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)

## Project Overview

This project focuses on implementing L2 regularization and dropout techniques in neural networks, aimed at improving model generalization and preventing overfitting. It covers functions for calculating costs with L2 regularization, updating weights with gradient descent, performing forward propagation with dropout, and incorporating early stopping in training.

## Prerequisites

- Python 3.x
- TensorFlow
- Keras
- NumPy

## Task Summaries

0. **L2 Regularization Cost**:  
   Calculates the total cost of a neural network, including L2 regularization, based on the given cost, regularization parameter, weights, number of layers, and data points.

1. **Gradient Descent with L2 Regularization**:  
   Updates the weights and biases of a neural network using gradient descent while incorporating L2 regularization, adjusting for the learning rate and regularization parameter.

2. **L2 Regularization Cost (Keras)**:  
   Computes the total cost of a Keras model with L2 regularization applied, based on the cost tensor of the network without regularization.

3. **Create a Layer with L2 Regularization**:  
   Creates a neural network layer in TensorFlow with L2 regularization, specifying the number of nodes, activation function, and regularization parameter.

4. **Forward Propagation with Dropout**:  
   Conducts forward propagation using dropout regularization, returning the outputs of each layer and the dropout masks applied.

5. **Gradient Descent with Dropout**:  
   Updates the weights of a neural network using gradient descent while considering dropout regularization, incorporating the learning rate and probability of keeping nodes.

6. **Create a Layer with Dropout**:  
   Constructs a neural network layer with dropout in TensorFlow, specifying the number of nodes, activation function, and whether the model is in training mode.

7. **Early Stopping**:  
   Determines if gradient descent should be halted based on the validation cost, optimal cost, threshold, patience count, and the current count of iterations without improvement.
