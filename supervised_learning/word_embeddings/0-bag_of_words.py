#!/usr/bin/env python3
"""
creates a bag of words embedding matrix
"""
import string


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
        # create a set of unique words (features) from all the sentences
        #   by splitting each sentence into words and adding them to the set,
        #   ensuring each word is included only once
        features = set(
            word for sentence in normalizedSentences
            for word in sentence.split()
        )
    else:
        features = vocab  # if vocab is not None, set it as the features

    # convert the set to a list to have a consistent order
    features = sorted(features)

    embeddings = []

    # loop through each normalized sentence
    for sentence in normalizedSentences:
        # split the sentence into individual words
        sentenceWords = sentence.split()

        # create a vector for a sentence where each element counts
        #   how many times a vocab word appears in the sentence
        sentenceVectorized = [sentenceWords.count(word) for word in features]

        # add created vector to the list of all embeddings
        embeddings.append(sentenceVectorized)

    return embeddings, features
