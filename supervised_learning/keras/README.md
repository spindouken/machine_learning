# Keras (Tensorflow 2)

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)

## Project Overview

This project involves building, training, and managing neural networks using the Keras library. It covers constructing models, optimizing training parameters, implementing regularization techniques, saving/loading models and configurations, and evaluating model performance.

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy

## Task Summaries

0. **Sequential**:  
   Builds a Keras neural network model without using the Sequential class, specifying input features, layer configurations, activation functions, L2 regularization, and dropout probability.

1. **Input**:  
   Constructs a Keras model similarly to the Sequential task but explicitly without using the Sequential class.

2. **Optimize**:  
   Configures Adam optimization for a Keras model, setting categorical crossentropy as the loss function and accuracy as the evaluation metric.

3. **One Hot**:  
   Converts a label vector into a one-hot encoded matrix, ensuring the last dimension matches the number of classes.

4. **Train**:  
   Trains a Keras model using mini-batch gradient descent with specified input data, labels, batch size, and epochs, returning the training history.

5. **Validate**:  
   Updates the training function to include validation data analysis, assessing model performance on validation sets.

6. **Early Stopping**:  
   Enhances the training function to incorporate early stopping based on validation loss, applying a patience parameter for improved control.

7. **Learning Rate Decay**:  
   Modifies the training function to implement learning rate decay based on epoch progression, printing updates on learning rate changes.

8. **Save Only the Best**:  
   Updates the training function to save the model only if it achieves the best validation loss during training.

9. **Save and Load Model**:  
   Implements functions to save an entire Keras model to a specified file and load a model from a given file.

10. **Save and Load Weights**:  
    Provides functionality to save and load a model's weights in specified formats.

11. **Save and Load Configuration**:  
    Implements functions to save and load a model's configuration in JSON format, facilitating easy reconfiguration.

12. **Test**:  
    Tests a neural network model on specified data, returning the loss and accuracy metrics while optionally printing results.

13. **Predict**:  
    Makes predictions using a trained neural network model on provided input data, with options for verbosity in output.

