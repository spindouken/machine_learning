#!/usr/bin/env python3
"""creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    creates a confusion matrix
    labels: one-hot numpy.ndarray of shape (m, classes)
        containing the correct labels for each data point
        m: number of data points
        classes: number of classes
    logits: one-hot numpy.ndarray of shape (m, classes)
        containing the predicted labels
    Returns a confusion numpy.ndarray of shape (classes, classes)
    """
    labelsIndices = np.argmax(labels, axis=1)
    logitsIndices = np.argmax(logits, axis=1)

    classesCount = labels.shape[1]

    confusionMatrix = np.zeros((classesCount, classesCount))

    for i in range(labels.shape[0]):
        confusionMatrix[labelsIndices[i], logitsIndices[i]] += 1

    return confusionMatrix
