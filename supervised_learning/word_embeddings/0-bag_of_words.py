#!/usr/bin/env python3
"""
creates a bag of words embedding matrix
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    creates a bag of words embedding matrix
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
    # condition to check if vocab is None, if not None, use vocab words
    #     if None, use all words
    if vocab is not None:
        countVectorizer = CountVectorizer(vocabulary=vocab)
    else:
        countVectorizer = CountVectorizer()
    # fit model to the data (sentences), then transform it into
    #     a bag of words embedding matrix
    embeddings = countVectorizer.fit_transform(sentences).toarray()
    # get the feature names (words) [feature names will be columns of matrix]
    features = countVectorizer.get_feature_names_out()

    return embeddings, features
