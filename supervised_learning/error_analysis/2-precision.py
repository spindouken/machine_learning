#!/usr/bin/env python3
"""
calculates the precision for each class in a confusion matrix
"""
import numpy as np


def precision(confusion):
    """
    confusion (numpy.ndarray): confusion matrix
        of shape (classes, classes)
    Returns numpy.ndarray: array of shape (classes,)
        containing the precision of each class
    """
    truePositives = np.diag(confusion)

    truePositives_plus_falsePositives = np.sum(confusion, axis=0)

    precision = truePositives / truePositives_plus_falsePositives

    return precision
