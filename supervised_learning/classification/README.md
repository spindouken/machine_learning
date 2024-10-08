# Neural Network Implementation for Binary Classification

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)

## Project Overview

This project involved building a neural network system for binary classification from scratch. The development progressed from a simple neuron model to a deep neural network with multiple layers, focusing on implementing key neural network components such as forward propagation, cost functions, gradient descent, and training methods. Additional features like verbosity, graphing options, and error handling were added to enhance model training and evaluation.

The files build on each other starting with 0. The main files import and implement/test each of their associated numbered files (ex. 0-main.py imports and tests 0-neuron.py)

## Key Features

- **Single Neuron Model**: Implements a perceptron model for binary classification with forward propagation, cost function, and gradient descent training.
- **Neural Network with One Hidden Layer**: A multi-layer perceptron (MLP) with one hidden layer, handling forward propagation, backpropagation, and efficient training.
- **Deep Neural Network**: Implements a deep neural network using multiple layers, enabling more complex binary classification tasks.
- **Training Enhancements**: Includes verbosity and graphing options for real-time monitoring of model training, displaying cost reduction over iterations.
- **Model Evaluation**: Provides methods for evaluating the model's performance, returning predictions and cost.

## Prerequisites

To run this project, the following software and libraries are required:

- Python 3.x
- Numpy
- Matplotlib (for graphing cost during training)
- Jupyter Notebook (optional, for running the code interactively)

## File Summaries

**0-neuron.py**:  
   Created a `Neuron` class for binary classification, initializing weights and bias, with error handling for input features.

**1-neuron.py**:  
   Privatized the `Neuron` class attributes (weights, bias, activated output) and provided getter methods for these values.

**2-neuron.py**:  
   Added a `forward_prop` method to the `Neuron` class to compute forward propagation using a sigmoid function, updating and returning the activated output.

**3-neuron.py**:  
   Implemented a `cost` method in the `Neuron` class to calculate the logistic regression cost, ensuring numerical stability.

**4-neuron.py**:  
   Created an `evaluate` method in the `Neuron` class to assess the predictions and return both predictions and cost.

**5-neuron.py**:  
   Added a `gradient_descent` method to perform gradient descent and update the weights and bias of the neuron.

**6-neuron.py**:  
   Introduced a `train` method to the `Neuron` class for iterative training using gradient descent with error handling for input parameters.

**7-neuron.py**:  
   Enhanced the `train` method to include verbosity and graphing options to visualize training progress and cost reduction over time.

**8-neural_network.py**:  
   Defined a `NeuralNetwork` class with one hidden layer, initializing weights and biases for both layers.

**9-neural_network.py**:  
   Privatized the attributes of the `NeuralNetwork` class and provided getter methods to access them.

**10-neural_network.py**:  
    Implemented a `forward_prop` method to compute forward propagation through both layers of the neural network.

**11-neural_network.py**:  
    Added a `cost` method to compute the cost of the neural network’s predictions, based on logistic regression.

**12-neural_network.py**:  
    Created an `evaluate` method to return both predictions and cost for a given input dataset.

**13-neural_network.py**:  
    Implemented a `gradient_descent` method to update the weights and biases of both layers using backpropagation.

**14-neural_network.py**:  
    Introduced a `train` method to manage the training process over multiple iterations with appropriate error handling.

**15-neural_network.py**:  
    Enhanced the `train` method by adding options for verbosity and graphing, allowing real-time monitoring and visualization of the training process.

**16-deep_neural_network.py**:  
    Created a `DeepNeuralNetwork` class for binary classification, initializing input features and layers with He initialization and validation.

**17-deep_neural_network.py**:  
    Privatized the attributes of the `DeepNeuralNetwork` class, including `L`, `cache`, and `weights`, and added getter methods.

**18-deep_neural_network.py**:  
    Added a `forward_prop` method to compute forward propagation for the deep neural network and update the cache.

**19-deep_neural_network.py**:  
    Implemented a `cost` method to calculate the logistic regression cost for the deep neural network.

**20-deep_neural_network.py**:  
    Created an `evaluate` method to return both predictions and cost for the deep neural network.

**21-deep_neural_network.py**:  
    Added a `gradient_descent` method for backpropagation, updating weights for each layer of the deep neural network.

**22-deep_neural_network.py**:  
    Introduced a `train` method for training the deep neural network, managing iterations and updating parameters with error handling.

**23-deep_neural_network.py**:  
    Enhanced the `train` method with verbosity and graphing options for monitoring the deep neural network’s performance during training.

**24-one_hot_encode.py**:  
    Implemented a function to perform one-hot encoding for classification labels.

**25-one_hot_decode.py**:  
    Created a function to decode one-hot encoded labels back into their original form.

**26-deep_neural_network.py**:  
    Further optimized the `DeepNeuralNetwork` class by adding additional layer functions for flexible architectures.

**27-deep_neural_network.py**:  
    Improved performance of the `DeepNeuralNetwork` by implementing more efficient matrix operations during forward propagation.

**28-deep_neural_network.py**:  
    Finalized the `DeepNeuralNetwork` implementation with hyperparameter tuning options and further enhancements for training stability.
