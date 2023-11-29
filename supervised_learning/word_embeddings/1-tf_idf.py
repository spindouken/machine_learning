#!/usr/bin/env python3
"""
creates a tf-idf embedding
"""
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    creates a tf-idf embedding
    sentences is a list of sentences to analyze
    vocab is a list of the vocabulary words to use for the analysis
        If None, all words within sentences should be used
    Returns: embeddings, features
        embeddings is a numpy.ndarray of shape (s, f)
            containing the embeddings
            s is the number of sentences in sentences
            f is the number of features analyzed
        features is a list of the features used for embeddings
    """
    # normalize sentences: lowercase, remove punctuation,
    #   and handle possessives before splitting the sentences into words
    normalizedSentences = [
        sentence.lower()
        .replace("'s", "")
        .translate(str.maketrans("", "", string.punctuation))
        for sentence in sentences
    ]

    # use all words in the sentences if vocab is not provided
    if vocab is None:
        # Create a set of unique words from all sentences if vocab not provided
        features = sorted(
            set(word for sentence in normalizedSentences
                for word in sentence.split())
        )
    else:
        features = sorted(vocab)  # if vocab is not None, set as the features

    embedding = (
        TfidfVectorizer(vocabulary=features)
        .fit_transform(normalizedSentences)
        .toarray()
    )

    return embedding, features
