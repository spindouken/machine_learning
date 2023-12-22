#!/usr/bin/env python3
"""   performs semantic search on a corpus of documents"""
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os


def semantic_search(corpus_path, sentence):
    """
    corpus_path is the path to the corpus of reference documents on which
    to perform semantic search
    sentence is the sentence from which to perform semantic search

    Returns: the reference text of the document most similar to sentence
    """
    # include the query sentence in the documents list
    documents = [sentence]

    # read documents from the corpus path
    for filename in os.listdir(corpus_path):
        if filename.endswith(".md"):
            with open(os.path.join(corpus_path, filename), "r", encoding="utf-8") as f:
                documents.append(f.read())

    # load the semantic search model
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    # create embeddings for the documents
    embeddings = model(documents)

    # calculate the correlation matrix for the embeddings
    correlationMatrix = np.inner(embeddings, embeddings)

    # find the index of the document most similar to the input sentence
    closestDocIndex = np.argmax(correlationMatrix[0, 1:])

    # retrieve the reference text of the document most similar to input sentence
    mostSimilarDoc = documents[closestDocIndex + 1]

    return mostSimilarDoc
