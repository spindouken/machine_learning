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
    import tensorflow.keras as K
    from gensim.models import Word2Vec

    if not isinstance(model, Word2Vec):
        raise ValueError("input must be a trained gensim word2vec model")

    # extract embeddings from trained input gensim word2vec model
    embeddings = model.wv.vectors

    # extract the number of tokens and the dimensionality of the embeddings
    #   from the embeddings
    numTokens, embeddingDim = embeddings.shape

    # create and return trainable keras embedding layer
    return K.layers.Embedding(input_dim=numTokens,
                              output_dim=embeddingDim,
                              weights=[embeddings],
                              trainable=True)
