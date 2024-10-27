# Convolutional Neural Networks

## Supplementary Medium Article
https://medium.com/@masonthecount/imagenet-classification-with-deep-convolutional-neural-networks-6c35923c0545

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)

## Project Overview

This project covers the implementation of forward and backward propagation in convolutional and pooling layers of neural networks, along with the construction of the LeNet-5 architecture using both TensorFlow and Keras. Key concepts include convolution operations, pooling techniques, backpropagation, and model building.

## Prerequisites

- Python 3.x
- TensorFlow
- Keras
- NumPy

## Task Summaries

0. **Convolutional Forward Prop**:  
   Implements a function that performs forward propagation over a convolutional layer. It takes inputs from the previous layer, kernels, biases, an activation function, padding type, and stride. Utilizes NumPy for array manipulations.

1. **Pooling Forward Prop**:  
   Creates a function for forward propagation in a pooling layer, allowing for either max or average pooling. Inputs include the previous layer's output, kernel shape, stride, and mode. Uses NumPy for handling multidimensional arrays.

2. **Convolutional Back Prop**:  
   Develops a function to perform backpropagation in a convolutional layer. It computes gradients for the previous layer's output, kernels, and biases based on the derivatives provided. Uses NumPy for array operations.

3. **Pooling Back Prop**:  
   Implements a function for backpropagation in a pooling layer, calculating gradients with respect to the previous layer's output. Accepts inputs such as the derivatives, previous layer's output, kernel shape, stride, and mode. Utilizes NumPy for numerical computations.

4. **LeNet-5 (TensorFlow 1)**:  
   Builds a modified LeNet-5 architecture using TensorFlow, defining the structure of the neural network with convolutional, pooling, and fully connected layers. Initializes weights with the he_normal method and specifies activation functions. Returns outputs, loss, and accuracy metrics.

5. **LeNet-5 (Keras)**:  
   Constructs a modified LeNet-5 model using Keras. Defines the model architecture, initializes weights with the he_normal method, and ensures reproducibility with a seed. Compiles the model to use Adam optimization and includes accuracy metrics.
