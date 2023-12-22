#!/usr/bin/env python3
"""
answers questions from a reference text
"""

ender = 1

while ender == 1:
    """it does the thing, look at it go"""
    question = input("Q: ")
    exitWords = ["exit", "quit", "goodbye", "bye"]

    if question.lower().strip() in exitWords:
        print("A: Goodbye")
        ender = 0
    else:
        print("A: ")
