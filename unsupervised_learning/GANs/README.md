# Generative Adversarial Networks (From Scratch Using Tensorflow)
## Overview

This repository contains from scratch implementations of Generative Adversarial Networks (GANs) focused on image generation, featuring both a standard Deep Convolutional GAN (DCGAN) and a Progressive Growing GAN. The projects leverage TensorFlow to build and train models capable of generating high-quality images from random noise, specifically utilizing the MNIST dataset for initial training and advanced datasets for further experimentation.

## Projects

### 1. dcgan/MNIST_DCGAN.py

#### Project Summary

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate images from random noise using the MNIST dataset. The architecture is designed to learn the underlying distribution of handwritten digits, enabling the model to create new digit images that resemble the original dataset.

Key features include:

- **DCGAN Architecture**: Comprises a generator and a discriminator working in tandem to produce and evaluate images, respectively.
- **Image Normalization**: Normalizes MNIST images to a range of [-1, 1] for better convergence during training.
- **Loss Function and Optimization**: Employs binary cross-entropy as the loss function and the Adam optimizer for training efficiency.
- **Real-time Image Generation**: Generates and saves images at specified epochs to monitor training progress visually.
- **Integration with Weights and Biases (WandB)**: Optional integration for real-time experiment tracking and logging of hyperparameters.

#### Goals

- **Data Preparation**: Load and preprocess the MNIST dataset, normalizing the pixel values.
- **Build Generator and Discriminator**: Create a generator model using transposed convolutional layers and a discriminator model with convolutional layers.
- **Image Generation and Saving**: Implement a function to generate and save images at specified epochs.
- **Hyperparameter Management**: Set batch size, epochs, and noise dimensions for training.
- **Loss and Optimization**: Define the binary cross-entropy loss and initialize the Adam optimizer.
- **Training Loop**: Establish a training loop for iterating through the dataset and applying gradients.
- **Tracking and Logging**: Utilize WandB for experiment tracking or print metrics to the console.
- **Visualization**: Continuously visualize the generator's output and monitor changes in image quality.

---

### 2. PROGRESSIVEDCGAN.py

#### Project Summary

The Progressive Growing GAN (ProGAN) implementation builds upon the principles of GANs by introducing a progressive training technique. This method starts with low-resolution images and gradually increases the resolution as training progresses, allowing the network to learn coarse features before fine details. The architecture is adapted for handling the complexities of high-resolution image generation, enhancing the overall quality of generated outputs.

Key features include:

- **Progressive Training**: Utilizes a gradual approach to increase image resolution, enhancing training stability and image quality.
- **Layer Modification**: Dynamically adds layers to both the generator and discriminator during training, adapting to increasing resolution.
- **Fade-in Technique**: Implements a fade-in strategy for new layers to ensure smooth transitions during training.
- **Custom Models**: Defines custom generator and discriminator classes for better control over the architectures.
- **Real-time Image Generation**: Similar to the DCGAN, generates and saves images at specified epochs.

#### Goals

- **Data Preparation**: Load and preprocess the MNIST dataset for training.
- **Build Initial Generator and Discriminator**: Create initial models for both the generator and discriminator.
- **Dynamic Layer Addition**: Implement functions to add layers progressively to the models.
- **Training Step Logic**: Develop a training step that incorporates the fade-in technique and dynamic updates.
- **Image Generation**: Save generated images at specified epochs for visualization.
- **Track Performance**: Monitor training metrics and adjust hyperparameters for optimal performance.

## Installation

To run these projects, ensure you have TensorFlow and the necessary dependencies installed. You can install them using pip:

```bash
pip install tensorflow matplotlib tqdm
