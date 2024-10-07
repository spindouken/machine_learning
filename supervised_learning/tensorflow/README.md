# Neural Network Training with TensorFlow

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)

## Project Overview

This project involved building and training a neural network using TensorFlow. The primary goals were to implement essential concepts such as creating placeholders, defining layers, executing forward propagation, and managing training operations. It focused on evaluating model performance through accuracy and loss calculations.

## Key Features

- **Placeholders**: Utilized TensorFlow placeholders for flexible input data and one-hot encoded label management during training.
- **Layer Creation**: Developed a function to construct neural network layers, specifying the number of nodes and activation functions while using He initialization for weights to enhance training efficiency.
- **Forward Propagation**: Built the forward propagation graph, connecting layers to enable the flow of data through the network, facilitating the transformation from input to output.
- **Accuracy Calculation**: Implemented a method to compute prediction accuracy by comparing the model's outputs to actual labels, which aids in assessing model performance.
- **Loss Function**: Defined a function to compute softmax cross-entropy loss, providing a measure of the discrepancy between predicted outputs and actual labels, essential for model training.
- **Training Operation**: Created a training operation that employs gradient descent for optimizing network parameters based on the calculated loss.
- **Model Training**: Managed the training process using provided datasets, capturing training and validation costs and accuracies at specified intervals, with functionality for model saving.
- **Model Evaluation**: Evaluated the trained model on test data, returning predictions, accuracy, and loss, while enabling model loading from a specified path for testing.

## Prerequisites

To run this project, the following software and libraries are required:

- Python 3.x
- TensorFlow (version compatible with the project)
- NumPy
- Matplotlib (optional, for visualizations)

## Task Summaries

0. **Placeholders**:
   Created placeholders for input data and one-hot encoded labels using TensorFlow.

1. **Layers**:
   Implemented a function to create neural network layers with specified nodes and activation functions, incorporating He initialization for weights.

2. **Forward Propagation**:
   Built the forward propagation graph for the neural network by connecting layers based on specified sizes and activation functions.

3. **Accuracy**:
   Calculated prediction accuracy by comparing the predicted outputs with actual labels, returning a tensor with the accuracy value.

4. **Loss**:
   Defined a function to compute softmax cross-entropy loss between predicted outputs and actual labels, returning the loss value as a tensor.

5. **Train Operation**:
   Created a training operation utilizing gradient descent to optimize the network based on the calculated loss and a given learning rate.

6. **Train**:
   Constructed and trained the neural network using provided training and validation datasets, outputting training and validation costs and accuracies at specified iterations.

7. **Evaluate**:
   Evaluated the trained model on test data, returning predictions, accuracy, and loss while loading the model from a specified path.
