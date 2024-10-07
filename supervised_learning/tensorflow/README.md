# Neural Network Training with TensorFlow

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)

## Project Overview

This project focuses on building and training a neural network using TensorFlow. It covers essential concepts in neural network architecture, including placeholders, layers, forward propagation, loss calculation, accuracy assessment, training operations, model training, and evaluation. By implementing a series of tasks, you'll gain hands-on experience in constructing a neural network and understanding its components.

## Key Features

- **Placeholders**: Define inputs and outputs using TensorFlow's placeholder functionality.
- **Custom Layers**: Create neural network layers with specific configurations and activation functions.
- **Forward Propagation**: Build a forward propagation graph to enable data flow through the network.
- **Accuracy Measurement**: Implement a method to calculate the prediction accuracy against true labels.
- **Loss Function**: Utilize softmax cross-entropy loss for effective training.
- **Training Operations**: Optimize network weights using gradient descent.
- **Model Training**: Train the neural network on provided datasets, monitoring performance metrics.
- **Model Evaluation**: Assess the trained model on unseen data to evaluate its performance.

## Prerequisites

To run this project, ensure you have the following installed:

- Python 3.x
- TensorFlow (version 2.x recommended)
- NumPy
- Matplotlib (for visualization)

## Task Summaries

1. **Placeholders**:  
   Create two placeholders, `x` for input data and `y` for one-hot encoded labels, using TensorFlow.

2. **Layers**:  
   Implement a function to create a neural network layer, specifying the number of nodes and activation function, while using He initialization for weights.

3. **Forward Propagation**:  
   Build the forward propagation graph for the neural network by connecting layers based on specified sizes and activation functions.

4. **Accuracy**:  
   Calculate the prediction accuracy by comparing the predicted outputs with the actual labels, returning a tensor with the accuracy value.

5. **Loss**:  
   Define a function to compute the softmax cross-entropy loss between predicted outputs and actual labels, returning the loss value as a tensor.

6. **Train Operation**:  
   Create a training operation that utilizes gradient descent to optimize the network based on the calculated loss and a given learning rate.

7. **Train**:  
   Construct and train the neural network using provided training and validation datasets, outputting training and validation costs and accuracies at specified iterations, and saving the trained model.

8. **Evaluate**:  
   Evaluate the trained model on test data, returning predictions, accuracy, and loss while loading the model from a specified path.
