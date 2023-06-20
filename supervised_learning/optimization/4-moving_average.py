#!/usr/bin/env python3
"""calculates the weighted moving average of a data set"""


def moving_average(data, beta):
    """
    data: list of data to calculate the moving average of
    beta: is the weight used for the moving average
    Returns a list containing the moving averages of data
    """
    movingAverages = []
    bias_correction = 0
    for i in range(len(data)):
        bias_correction = ((bias_correction * beta) + ((1 - beta) * data[i]))
        movingAveragesCorrected = bias_correction / (1 - beta ** (i + 1))
        movingAverages.append(movingAveragesCorrected)

    return movingAverages
