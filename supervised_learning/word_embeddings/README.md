# Natural Language Processing - Word Embeddings

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)

## Project Overview

This project explores various word embedding techniques utilized in natural language processing (NLP). It covers methods such as Bag of Words, TF-IDF, Word2Vec, FastText, and ELMo. The focus is on implementing these models, converting them for use in Keras, and understanding their applications in text representation and vectorization.

## Key Features

- **Multiple Embedding Techniques**: Implements a range of techniques for text representation, allowing for comparison and evaluation.
- **Keras Integration**: Converts popular models like Word2Vec and FastText into Keras-compatible formats for further training.
- **Comprehensive Approach**: Covers both traditional methods (Bag of Words, TF-IDF) and advanced neural models (Word2Vec, FastText, ELMo).

## Prerequisites

- Python
- TensorFlow
- Gensim
- NumPy
- Pandas

## File Summaries

0. **Bag Of Words**: 
   - File: `0-bag_of_words.py`
   - This script implements the `bag_of_words` function from scratch to create a Bag of Words embedding matrix from a list of sentences. It normalizes the sentences by converting them to lowercase, removing punctuation, and handling possessives. If no vocabulary is provided, it generates a set of unique words from the sentences. The function returns a numpy array of embeddings and a list of features.

1. **TF-IDF**: 
   - File: `1-tf_idf.py`
   - This script implements the `tf_idf` function, which creates a TF-IDF embedding matrix. It also normalizes sentences similarly to the Bag of Words implementation. It uses the `TfidfVectorizer` from the sklearn library to compute the TF-IDF values. The function returns a numpy array of embeddings and a list of features.

2. **Train Word2Vec**: 
   - File: `2-word2vec.py`
   - This script contains the `word2vec_model` function that builds and trains a Word2Vec model using Gensim. The model is trained using parameters such as vector size, window size, and training type (CBOW or Skip-gram). The training is conducted using the Gensim library functions, and the trained model is returned.

3. **Extract Word2Vec**: 
   - File: `3-gensim_to_keras.py`
   - This script implements the `gensim_to_keras` function that converts a trained Gensim Word2Vec model into a Keras Embedding layer. It utilizes TensorFlow's Keras library to create the embedding layer and extracts the trained word vectors from the Gensim model. The resulting layer is trainable, allowing for further fine-tuning during model training.

4. **FastText**: 
   - File: `4-fasttext.py`
   - This script implements the `fasttext_model` function that creates and trains a FastText model using Gensim. It takes a list of sentences and various parameters such as size, min_count, and window size to configure the model. The training process builds the vocabulary first and then trains the model using the provided sentences. The function returns the trained FastText model.

5. **ELMo**: Complete a text file, `5-elmo`, by indicating the correct aspects of ELMo embedding training regarding internal weights, character embedding layers, and hidden states.
