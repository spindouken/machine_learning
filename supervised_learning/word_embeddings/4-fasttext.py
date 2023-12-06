#!/usr/bin/env python3
"""
creates and trains a gensim fastText model
"""


def fasttext_model(
    sentences,
    size=100,
    min_count=5,
    negative=5,
    window=5,
    cbow=True,
    iterations=5,
    seed=0,
    workers=1,
):
    """
    sentences is a list of sentences to be trained on
    size is the dimensionality of the embedding layer
    min_count is the minimum number of occurrences of
        a word for use in training
    window is the maximum distance between the
        current and predicted word within a sentence
    negative is the size of negative sampling
    cbow is a boolean to determine the training type;
        True is for CBOW; False is for Skip-gram
    iterations is the number of iterations to train over
    seed is the seed for the random number generator
    workers is the number of worker threads to train the model
    Returns: the trained model
    """
    from gensim.models import FastText

    # if cbow boolean input variable is set to True, training is CBOW
    if cbow is True:
        yaya = 0
    else:
        # if cbow parameter input is not boolean True,
        #   training type is skip-gram
        yaya = 1

    # compile gensim fasttext model using function paramter inputs
    model = FastText(
        vector_size=size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=yaya,
        epochs=iterations,
        seed=seed,
        workers=workers
        )

    # must build vocab before training
    model.build_vocab(corpus_iterable=sentences)

    # train gensim fastText model
    model.train(
        corpus_iterable=sentences,
        total_examples=len(sentences),
        epochs=model.epochs
        )

    # return trained gensim fastText model
    return model
