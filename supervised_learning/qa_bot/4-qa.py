#!/usr/bin/env python3
"""
answers questions from multiple reference texts
"""
import tensorflow as tf
import tensorflow_hub
from transformers import BertTokenizer


def question_answer(corpus_path):
    """
    reference is the reference text

    takes user input from CLI, initializes it as a question,
        formats it as lowercase with .lower(),
        strips any leading or trailing whitespace with .strip(),
        and checks to see if it is in the `exitWords` list
    If the question is in the `exitWords` list,
        responds with "Goodbye"
    If the question is not in the `exitWords` list,
        calls the `question_answer` function

    If the answer cannot be found in the reference text,
        responds with "Sorry, I do not understand your question."
    """
    ender = 1
    while ender == 1:
        # takes user input from CLI and identifies it as the question
        question = input("Q: ")
        exitWords = ["exit", "quit", "goodbye", "bye"]

        # process user input to account for character cases and whitespaces,
        #   then check if it is in the exitWords list
        if question.lower().strip() in exitWords:
            print("A: Goodbye")
            # if it is in the list, ender counter will go to zero, breaking the loop
            ender = 0
        else:
            reference = semantic_search(corpus_path, question)
            # if the user input was not an exit word, call question_answer function
            answer = answerFinder(question, reference)
            if answer is None:
                answer = "Sorry, I do not understand your question."
            # provide answer to user in CLI
            print("A: {}".format(answer))


def answerFinder(question, reference):
    """
    finds a snippet of text within a reference document to answer a question

    question is a string containing the question to answer
    reference is a string containing the reference document from which to find
        the answer

    Returns: a string containing the answer
    If no answer is found, return None
    If multiple answers are found, return the first one
    """
    # initialize a pre-trained BertTokenizer for tokenizing the input question and reference
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )

    # initialize the bert-uncased-tf2-qa model from the tensorflow-hub library
    model = tensorflow_hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # tokenize the question
    questionTokenized = tokenizer.tokenize(question)
    # tokenize the reference
    referenceTokenized = tokenizer.tokenize(reference)

    # Combine tokens with special tokens
    tokens = ["[CLS]"] + questionTokenized + ["[SEP]"] + referenceTokenized + ["[SEP]"]

    # Convert tokens to IDs
    inputWord_IDs = tokenizer.convert_tokens_to_ids(tokens)

    # Create input mask
    inputMask = [1] * len(inputWord_IDs)

    # Create input type IDs
    segmentIDs = [0] * (1 + len(questionTokenized) + 1) + [1] * (
        len(referenceTokenized) + 1
    )

    # Prepare TensorFlow tensors
    inputWord_IDs = tf.expand_dims(
        tf.convert_to_tensor(inputWord_IDs, dtype=tf.int32), 0
    )
    inputMask = tf.expand_dims(tf.convert_to_tensor(inputMask, dtype=tf.int32), 0)
    segmentIDs = tf.expand_dims(tf.convert_to_tensor(segmentIDs, dtype=tf.int32), 0)

    # Get model outputs
    outputs = model([inputWord_IDs, inputMask, segmentIDs])

    # Find start and end of answer
    answerStart = tf.argmax(outputs[0][0][1:]) + 1
    answerEnd = tf.argmax(outputs[1][0][1:]) + 1

    # Extract answer tokens
    answerTokens = tokens[answerStart : answerEnd + 1]

    # Return answer string or None
    return tokenizer.convert_tokens_to_string(answerTokens) if answerTokens else None


import numpy as np
import os


def semantic_search(corpus_path, sentence):
    """
    performs semantic search on a corpus of documents

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
    model = tensorflow_hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    )

    # create embeddings for the documents
    embeddings = model(documents)

    # calculate the correlation matrix for the embeddings
    correlationMatrix = np.inner(embeddings, embeddings)

    # find the index of the document most similar to the input sentence
    closestDocIndex = np.argmax(correlationMatrix[0, 1:])

    # retrieve the reference text of the document most similar to input sentence
    mostSimilarDoc = documents[closestDocIndex + 1]

    return mostSimilarDoc
