#!/usr/bin/env python3
"""
creates and trains a gensim word2vec model
"""


def word2vec_model(sentences, size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   iterations=5, seed=0, workers=1):
    """
    creates and trains a gensim word2vec model

    sentences is a list of sentences to be trained on
    size is the dimensionality of the embedding layer
    min_count is the minimum number of occurrences of a word
        for use in training
    window is the maximum distance between the current
        and predicted word within a sentence
    negative is the size of negative sampling
        (for each positive sample, how many negative samples to draw)
    cbow is a boolean to determine the training type;
        True is for CBOW; False is for Skip-gram
    iterations is the number of iterations to train over
    seed is the seed for the random number generator
    workers is the number of worker threads to train the model
    Returns: the trained model
    """
    from gensim.models import Word2Vec

    # if cbow boolean input variable is set to True, training is CBOW
    if cbow is True:
        yaya = 0
    else:
        # if cbow parameter input is not boolean True,
        #   training type is skip-gram
        yaya = 1

    # compile word2vec model using function parameter inputs
    model = Word2Vec(
        sentences,
        vector_size=size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=yaya,
        seed=seed,
        workers=workers,
        )

    # train gensim word2vec model
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=iterations
        )

    # return trained gensim word2vec model
    return model
