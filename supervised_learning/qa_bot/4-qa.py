#!/usr/bin/env python3
"""
answers questions from multiple reference texts
"""
semantic_search = __import__('3-semantic_search').semantic_search
question_answer = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """
    answers questions from multiple reference texts
    """
    exit = ["exit", "quit", "goodbye", "bye"]
    while True:
        question = input("Q: ")
        if question.lower() in exit:
            print("A: Goodbye")
            break
        reference = semantic_search(corpus_path, question)
        answer = question_answer(question, reference)
        if answer is None or answer == "":
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))
