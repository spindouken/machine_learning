#!/usr/bin/env python3
"""
calculates the F1 score of a confusion matrix
"""
import numpy as np


def f1_score(confusion):
    """
    confusion (numpy.ndarray): confusion matrix
        of shape (classes, classes)
    Returns numpy.ndarray: array of shape (classes,)
        containing the F1 score of each class
    """
    sensitivityFunction = __import__('1-sensitivity').sensitivity
    precisionFunction = __import__('2-precision').precision

    precision = precisionFunction(confusion)
    sensitivity = sensitivityFunction(confusion)

    f1 = 2 * precision * sensitivity / (precision + sensitivity)

    return f1
