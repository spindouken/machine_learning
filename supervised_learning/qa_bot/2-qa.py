"""
answers questions from a reference text
"""
import tensorflow as tf
from transformers import BertTokenizer


def answer_loop(reference):
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
            # if the user input was not an exit word, call question_answer function
            answer = question_answer(question, reference)
            if answer is None:
                answer = "Sorry, I do not understand your question."
            # provide answer to user in CLI
            print("A: {}".format(answer))
