#!/usr/bin/env python3
"""
converts a gensim word2vec model to a keras Embedding layer
"""


def gensim_to_keras(model):
    """
    converts a gensim word2vec model to a keras Embedding layer

    model is a trained gensim word2vec models
    Returns: the trainable keras Embedding
    """
