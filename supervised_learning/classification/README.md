# Neural Network Implementation for Binary Classification

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)

## Project Overview

This project involved building a neural network system for binary classification from scratch. The development progressed from a simple neuron model to a deep neural network with multiple layers, focusing on implementing key neural network components such as forward propagation, cost functions, gradient descent, and training methods. Additional features like verbosity, graphing options, and error handling were added to enhance model training and evaluation.

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

## Task Summaries

0. **Task 0 (0-main.py, 0-neuron.py)**:  
   Created a `Neuron` class for binary classification, initializing weights and bias, with error handling for input features.

1. **Task 1 (1-main.py, 1-neuron.py)**:  
   Privatized the `Neuron` class attributes (weights, bias, activated output) and provided getter methods for these values.

2. **Task 2 (2-main.py, 2-neuron.py)**:  
   Added a `forward_prop` method to the `Neuron` class to compute forward propagation using a sigmoid function, updating and returning the activated output.

3. **Task 3 (3-main.py, 3-neuron.py)**:  
   Implemented a `cost` method in the `Neuron` class to calculate the logistic regression cost, ensuring numerical stability.

4. **Task 4 (4-main.py, 4-neuron.py)**:  
   Created an `evaluate` method in the `Neuron` class to assess the predictions and return both predictions and cost.

5. **Task 5 (5-main.py, 5-neuron.py)**:  
   Added a `gradient_descent` method to perform gradient descent and update the weights and bias of the neuron.

6. **Task 6 (6-main.py, 6-neuron.py)**:  
   Introduced a `train` method to the `Neuron` class for iterative training using gradient descent with error handling for input parameters.

7. **Task 7 (7-main.py, 7-neuron.py)**:  
   Enhanced the `train` method to include verbosity and graphing options to visualize training progress and cost reduction over time.

8. **Task 8 (8-main.py, 8-neural_network.py)**:  
   Defined a `NeuralNetwork` class with one hidden layer, initializing weights and biases for both layers.

9. **Task 9 (9-main.py, 9-neural_network.py)**:  
   Privatized the attributes of the `NeuralNetwork` class and provided getter methods to access them.

10. **Task 10 (10-main.py, 10-neural_network.py)**:  
    Implemented a `forward_prop` method to compute forward propagation through both layers of the neural network.

11. **Task 11 (11-main.py, 11-neural_network.py)**:  
    Added a `cost` method to compute the cost of the neural network’s predictions, based on logistic regression.

12. **Task 12 (12-main.py, 12-neural_network.py)**:  
    Created an `evaluate` method to return both predictions and cost for a given input dataset.

13. **Task 13 (13-main.py, 13-neural_network.py)**:  
    Implemented a `gradient_descent` method to update the weights and biases of both layers using backpropagation.

14. **Task 14 (14-main.py, 14-neural_network.py)**:  
    Introduced a `train` method to manage the training process over multiple iterations with appropriate error handling.

15. **Task 15 (15-main.py, 15-neural_network.py)**:  
    Enhanced the `train` method by adding options for verbosity and graphing, allowing real-time monitoring and visualization of the training process.

16. **Task 16 (16-main.py, 16-deep_neural_network.py)**:  
    Created a `DeepNeuralNetwork` class for binary classification with He initialization of weights for deep networks.

17. **Task 17 (17-main.py, 17-deep_neural_network.py)**:  
    Privatized the `DeepNeuralNetwork` attributes and ensured proper initialization with getter methods for the private attributes.

18. **Task 18 (18-main.py, 18-deep_neural_network.py)**:  
    Added a `forward_prop` method to perform forward propagation across multiple layers of the deep neural network.

19. **Task 19 (19-main.py, 19-deep_neural_network.py)**:  
    Implemented a `cost` method to calculate the cost of the deep neural network using logistic regression.

20. **Task 20 (20-main.py, 20-deep_neural_network.py)**:  
    Created an `evaluate` method for the deep neural network to return both predictions and cost for a given input dataset.

21. **Task 21 (21-main.py, 21-deep_neural_network.py)**:  
    Added a `gradient_descent` method to the deep neural network for updating weights and biases across all layers.

22. **Task 22 (22-main.py, 22-deep_neural_network.py)**:  
    Introduced a `train` method to manage training iterations for the deep neural network, with error handling and validation.

23. **Task 23 (23-main.py, 23-deep_neural_network.py)**:  
    Enhanced the `train` method with verbosity and graphing options for tracking performance during training.

24. **Task 24 (24-main.py, 24-one_hot_encode.py)**:  
    Implemented a one-hot encoding function to preprocess categorical data for the neural network.

25. **Task 25 (25-main.py, 25-one_hot_decode.py)**:  
    Developed a one-hot decoding function to convert predictions back into original label format.

26. **Task 26 (26-main.py, 26-deep_neural_network.py)**:  
    Improved the deep neural network training process with additional functionality for validation.

27. **Task 27 (27-main.py, 27-deep_neural_network.py)**:  
    Added model persistence to save and load trained deep neural networks for future use.

28. **Task 28 (28-main.py, 28-deep_neural_network.py)**:  
    Enhanced the deep neural network to support batch training for improved performance on large datasets.
