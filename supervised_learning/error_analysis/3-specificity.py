#!/usr/bin/env python3
"""
calculates the specificity for each class in a confusion matrix
"""
import numpy as np


def specificity(confusion):
    """
    confusion (numpy.ndarray): confusion matrix
        of shape (classes, classes)
    Returns numpy.ndarray: array of shape (classes,)
        containing the specificity of each class
    """
    total = np.sum(confusion)

    sum_row = np.sum(confusion, axis=1)
    sum_col = np.sum(confusion, axis=0)

    truePositives = np.diag(confusion)

    trueNegatives = total - sum_row - sum_col + truePositives

    falsePositives = sum_col - truePositives

    specificity = trueNegatives / (trueNegatives + falsePositives)

    return specificity
