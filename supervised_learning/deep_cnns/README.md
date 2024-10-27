# Deep Convolutional Architectures

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)

## Project Overview

This project focuses on building advanced neural network architectures using inception blocks, residual connections, and dense connections. Key concepts include constructing inception and identity blocks, projection blocks, and implementing DenseNet and ResNet architectures, with an emphasis on initialization techniques and activation functions.

## Prerequisites

- Python 3.x
- TensorFlow
- Keras
- NumPy

## Task Summaries

0. **Inception Block**:  
   Creates a function to build an inception block that combines multiple convolutional layers (1x1, 3x3, 5x5) with max pooling and applies ReLU activation. Returns the concatenated output of the inception block.

1. **Inception Network**:  
   Develops a function to build the inception network architecture, using the inception block. Assumes input shape of (224, 224, 3) and applies ReLU activation across all layers. Returns a Keras model.

2. **Identity Block**:  
   Implements a function for an identity block, featuring two 1x1 convolutions and a 3x3 convolution, all followed by batch normalization and ReLU activation. Initializes weights with he normal initialization. Returns the activated output of the block.

3. **Projection Block**:  
   Creates a function for a projection block that uses 1x1 and 3x3 convolutions with batch normalization and ReLU activation. Includes a shortcut connection and uses he normal initialization for weights. Returns the activated output.

4. **ResNet-50**:  
   Builds the ResNet-50 architecture utilizing identity and projection blocks, assuming input shape of (224, 224, 3). Applies batch normalization and ReLU activation to all convolutions and uses he normal initialization. Returns a Keras model.

5. **Dense Block**:  
   Implements a function to create a dense block that connects multiple convolutional layers using bottleneck layers. Applies batch normalization and ReLU activation, initializing weights with he normal initialization. Returns the concatenated output and the number of filters.

6. **Transition Layer**:  
   Creates a function for a transition layer that reduces the number of filters with compression. Applies batch normalization and ReLU activation, using he normal initialization. Returns the output and the number of filters.

7. **DenseNet-121**:  
   Develops a function to build the DenseNet-121 architecture, incorporating dense and transition layers. Assumes input shape of (224, 224, 3) and initializes weights with he normal initialization. Returns a Keras model.
