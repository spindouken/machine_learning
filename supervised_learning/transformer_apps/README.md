# Transformers

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [File Summaries](#file-summaries)

## Project Overview

This project involves building a machine translation system from Portuguese to English using a transformer model. It covers the creation of a dataset class for loading and preparing data, tokenizing sentences, encoding tokens, setting up a data pipeline, creating masks for attention mechanisms, and training a transformer model with appropriate optimizations and loss functions.

## Key Features

- **Dataset Preparation**
- **Token Encoding**
- **Optimized Data Pipeline**
- **Attention Mechanisms**
- **Transformer Training**

## Prerequisites

- Python
- TensorFlow (for building and training the model)
- NumPy (for numerical operations)

## File Summaries

0. **Dataset**: 
   - File: `0_dataset.py`
   - This file creates the `Dataset` class to load and prepare the Portuguese-English translation dataset. It initializes training and validation splits using TensorFlow datasets and sets up sub-word tokenizers using pre-trained models for both languages. The tokenizer setup is done using library functions, while dataset handling is implemented from scratch.

1. **Encode Tokens**: 
   - File: `1_encode.py`
   - This file enhances the `Dataset` class by adding the `encode` method, which encodes Portuguese and English sentences into token indices, including start and end tokens for both languages. The method utilizes basic string manipulations and indexing, implemented from scratch.

2. **TF Encode**: 
   - File: `2_tf_encode.py`
   - This file updates the `Dataset` class to include the `tf_encode` method, which serves as a TensorFlow wrapper around the `encode` method, ensuring proper tensor shape for training and validation data. The method integrates TensorFlow functions for tensor manipulation while the encoding logic is custom-built.

3. **Pipeline**: 
   - File: `3_pipeline.py`
   - This file modifies the `Dataset` class constructor to implement a data pipeline that filters, caches, shuffles, batches, and prefetches the training and validation datasets. The performance optimizations are achieved through a combination of TensorFlow functions and custom methods.

4. **Create Masks**: 
   - File: `4_create_masks.py`
   - This file implements the `create_masks` function to generate attention masks for the encoder and decoder, ensuring proper padding and masking for input and target sentences during training. The masking logic is developed from scratch, relying on basic array manipulations.

5. **Train**: 
   - File: `5_train.py`
   - This file develops the `train_transformer` function that builds and trains a transformer model for the translation task. It incorporates parameters for model architecture and training settings, utilizing Adam optimization, sparse categorical crossentropy loss, and scheduled learning rates. The training process includes custom logging of loss and accuracy metrics.
