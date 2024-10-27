# T4-10 RNNs From Scratch (RNN, GRU, LSTM, BRNN)

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)

## Project Overview

This project focuses on the from scratch implementation and forward propagation of various recurrent neural network (RNN) architectures, including basic RNN cells, Gated Recurrent Units (GRU), Long Short-Term Memory (LSTM) cells, deep RNNs, and bidirectional RNNs. Key concepts covered include the design of RNN components, forward propagation through time steps, and the integration of bidirectional processing in RNNs.

## Key Features

- **Custom RNN Architectures**: Implementation of various RNN types including GRU and LSTM.
- **Bidirectional Processing**: Ability to capture context from both directions in sequence data.
- **Deep RNN Support**: Capability to build RNNs with multiple layers for enhanced learning.

## Prerequisites

- Python
- NumPy

## Task Summaries

0. **RNN Cell**: Create the `RNNCell` class, which includes initialization of weights and biases for input, hidden state, and output dimensions, along with a method for forward propagation using softmax activation.

1. **RNN**: Implement the `rnn` function for forward propagation through a simple RNN, utilizing an `RNNCell` instance to process a sequence of data across multiple time steps.

2. **GRU Cell**: Develop the `GRUCell` class that represents a gated recurrent unit, initializing weights and biases for update, reset, and output gates, along with a method for forward propagation.

3. **LSTM Cell**: Create the `LSTMCell` class representing an LSTM unit, initializing weights and biases for forget, update, intermediate cell state, and output gates, including a method for forward propagation.

4. **Deep RNN**: Write the `deep_rnn` function to perform forward propagation through multiple layers of RNN cells, processing input data across time steps.

5. **Bidirectional Cell Forward**: Implement the `BidirectionalCell` class to represent a bidirectional RNN cell, initializing weights and biases for forward and backward hidden states and providing a method for forward propagation.

6. **Bidirectional Cell Backward**: Update the `BidirectionalCell` class by adding a method for backward propagation, calculating the previous hidden state from the next hidden state and input data.

7. **Bidirectional Output**: Enhance the `BidirectionalCell` class with an output method that computes all outputs from concatenated hidden states of both forward and backward directions.

8. **Bidirectional RNN**: Implement the `bi_rnn` function for forward propagation in a bidirectional RNN, utilizing the `BidirectionalCell` to process input data and return concatenated hidden states and outputs.
