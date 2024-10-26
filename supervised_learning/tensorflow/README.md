# Neural Network Training with TensorFlow

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)

## Project Overview

This project involves building and training a neural network using TensorFlow, focusing on concepts such as creating placeholders, layers, forward propagation, calculating accuracy and loss, training operations, model training, and evaluation. Key tasks include implementing functions for network architecture and training while managing model inputs and outputs.

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib (optional, for visualizations)

## Task Summaries

0. **Create Placeholders**:  
   Created placeholders for input data (`x`) and labels (`y`) for one-hot encoded representation using TensorFlow.

1. **Implement Layer Creation**:  
   Implemented a function to create a neural network layer, specifying the number of nodes and activation function, while using He initialization for weights.

2. **Build Forward Propagation**:  
   Built the forward propagation graph for the neural network by connecting layers based on specified sizes and activation functions.

3. **Calculate Accuracy**:  
   Calculated prediction accuracy by comparing the predicted outputs with the actual labels, returning a tensor with the accuracy value.

4. **Define Loss Function**:  
   Defined a function to compute the softmax cross-entropy loss between predicted outputs and actual labels, returning the loss value as a tensor.

5. **Create Training Operation**:  
   Created a training operation that utilizes gradient descent to optimize the network based on the calculated loss and a given learning rate.

6. **Train Model**:  
   Constructed and trained the neural network using provided training and validation datasets, outputting training and validation costs and accuracies at specified iterations, and saving the trained model.

7. **Evaluate Model**:  
   Evaluated the trained model on test data, returning predictions, accuracy, and loss while loading the model from a specified path.
