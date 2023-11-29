#!/usr/bin/env python3
"""
creates a tf-idf embedding
"""
import numpy as np
import string


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
        features = set(
            word for sentence in normalizedSentences
            for word in sentence.split()
        )
    else:
        features = vocab  # if vocab is not None, set it as the features

    # initialize tf (term frequency) matrix
    tf = []

    # loop through each normalized sentence
    for sentence in normalizedSentences:
        # split the sentence into individual words
        sentenceWords = sentence.split()

        # calculate term frequency
        sentenceTF = [
            sentenceWords.count(word) / len(sentenceWords) for word in features
        ]
        tf.append(sentenceTF)

    tf = np.array(tf)

    # calculate inverse document frequency
    idf = np.log(len(sentences) / (1 + np.sum(tf > 0, axis=0))) + 1

    # calculate tf-idf
    embeddings = tf * idf

    # normalize tf-idf embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.nan_to_num(embeddings)  # replace NaNs with zeros

    return embeddings, features
