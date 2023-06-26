#!/usr/bin/env python3
"""
calculates the sensitivity for each class in a confusion matrix:
"""
import numpy as np


def sensitivity(confusion):
    """
    confusion: (numpy.ndarray) confusion matrix
    of shape (classes, classes)
        row indices: re
        classes: number of classes
    Returns numpy.ndarray: array of shape (classes,) containing the sensitivity of each class
    """
    truePositives = np.diag(confusion)

    truePositives_plus_falseNegatives = np.sum(confusion, axis=1)

    sensitivity = truePositives / truePositives_plus_falseNegatives

    return sensitivity
