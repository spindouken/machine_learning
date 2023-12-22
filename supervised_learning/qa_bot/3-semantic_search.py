#!/usr/bin/env python3
"""
performs semantic search on a corpus of documents
"""
import os
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from transformers import BertTokenizer

def semantic_search(corpus_path, sentence):
    """
    performs semantic search on a corpus of documents
    corpus_path is the path to the corpus of reference documents on which
    to perform semantic search
    sentence is the sentence from which to perform semantic search
    Returns: the reference text of the document most similar to sentence
    """
    # load bert tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case=True
    )

    # load bert model
    model = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')

    # load corpus
    references = [sentence]
    for filename in os.listdir(corpus_path):
        if not filename.endswith('.md'):
            continue
        with open(corpus_path + '/' + filename, 'r', encoding='utf-8') as f:
            references.append(f.read())

    # tokenize
    tokens = tokenizer(references, padding=True, truncation=True, return_tensors='tf')

    # get embeddings
    embeddings = model(tokens)

    # get query embedding
    query = tf.constant([sentence])
    query_embedding = model(query)

    # compute dot product
    dot_product = tf.matmul(query_embedding, tf.transpose(embeddings))

    # compute softmax
    softmax = tf.nn.softmax(dot_product, axis=1)

    # get argmax
    argmax = tf.argmax(softmax, axis=1)

    # get result
    result = references[argmax.numpy()[0]]

    return result
