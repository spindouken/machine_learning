#!/usr/bin/env python3
"""
finds a snippet of text within a reference document to answer a question
"""
import tensorflow as tf
import tensorflow_hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    finds a snippet of text within a reference document to answer a question

    question is a string containing the question to answer
    reference is a string containing the reference document from which to find
        the answer

    utilizes the bert-uncased-tf2-qa model from the tensorflow-hub library
    uses the pretrained BertTokenizer, bert-large-uncased-whole-word-masking-finetuned-squad,
      from the transformers library

    BERT models don't understand text directly; they understand numeric representations.
      Each token is mapped to a unique numeric ID.
    BERT uses special tokens to understand the structure of the input.
      [CLS]: Classifier token (beginning of sentence)
      [SEP]: Separator token (end of each input)

    Returns: a string containing the answer
    If no answer is found, return None
    If multiple answers are found, return the first one
    """
    # initialize a pre-trained BertTokenizer object for tokenizing the input question and reference
    #   into a format suitable for the BERT model
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )

    # initialize the bert-uncased-tf2-qa model from the tensorflow-hub library
    model = tensorflow_hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # tokenize the question
    questionTokenized = tokenizer.tokenize(question)
    # tokenize the reference
    referenceTokenized = tokenizer.tokenize(reference)

    # combine input tokens with BERT special tokens CLS and SEP
    tokens = ["[CLS]"] + questionTokenized + ["[SEP]"] + referenceTokenized + ["[SEP]"]

    # converts the tokens into their corresponding numeric IDs
    inputWord_IDs = tokenizer.convert_tokens_to_ids(tokens)

    # create input mask
    inputMask = [1] * len(inputWord_IDs)

    # create segment IDs
    segmentIDs = [0] * (1 + len(questionTokenized) + 1) + [1] * (
        len(referenceTokenized) + 1
    )

    # create TensorFlow tensors
    inputWord_IDs = tf.expand_dims(
        tf.convert_to_tensor(inputWord_IDs, dtype=tf.int32), 0
    )
    inputMask = tf.expand_dims(tf.convert_to_tensor(inputMask, dtype=tf.int32), 0)
    segmentIDs = tf.expand_dims(tf.convert_to_tensor(segmentIDs, dtype=tf.int32), 0)

    # get model outputs
    outputs = model([inputWord_IDs, inputMask, segmentIDs])

    # find start and end of answer
    answerStart = tf.argmax(outputs[0][0][1:]) + 1
    answerEnd = tf.argmax(outputs[1][0][1:]) + 1

    # extract answer tokens
    answerTokens = tokens[answerStart : answerEnd + 1]

    # return answer string or None
    return tokenizer.convert_tokens_to_string(answerTokens) if answerTokens else None
