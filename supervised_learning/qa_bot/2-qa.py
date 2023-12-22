#!/usr/bin/env python3
"""
answers questions from a reference text
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

question_answer = __import__("0-qa").question_answer


def answer_loop(reference):
    """
    answers questions from a reference text

    reference is the reference text
    If the answer cannot be found in the reference text,
        respond with Sorry, I do not understand your question.
    """
    while True:
        question = input("Q: ")
        if question.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break
        answer = question_answer(question, reference)
        if answer is None:
            answer = "Sorry, I do not understand your question."
        print("A: {}".format(answer))
