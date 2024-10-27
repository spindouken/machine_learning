### Autoencoders (Built from Scratch Using TensorFlow)

#### Project Summary

The project involves building and training various types of autoencoders using TensorFlow. It covers the following concepts:

- Vanilla Autoencoder
- Sparse Autoencoder
- Convolutional Autoencoder
- Variational Autoencoder

#### File Summaries

0. **Vanilla Autoencoder**
- File: `0_vanilla.py`
    - Creates a basic autoencoder that takes input dimensions, hidden layer configurations, and latent dimensions. The model includes an encoder and decoder, using ReLU and sigmoid activations, compiled with Adam optimizer and binary cross-entropy loss.

1. **Sparse Autoencoder**
- File: `1_sparse.py`
    - Develops a sparse autoencoder by adding an L1 regularization parameter to the encoded output. Similar to the vanilla version, it uses specified input and latent dimensions, along with ReLU and sigmoid activations, and is compiled with the same optimizer and loss function.

2. **Convolutional Autoencoder**
- File: `2_convolutional.py`
    - Implements a convolutional autoencoder designed for image data. This model uses convolutional layers with specific kernel sizes, max pooling, and upsampling techniques, maintaining the structure of the encoder and decoder while applying ReLU and sigmoid activations.

3. **Variational Autoencoder**
- File: `3_variational.py`
    - Constructs a variational autoencoder that produces a latent representation along with the mean and log variance outputs. The model has distinct activation functions for different layers, compiled with Adam optimizer and binary cross-entropy loss, aimed at generating and reconstructing image data.
