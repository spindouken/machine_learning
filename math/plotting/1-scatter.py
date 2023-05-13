#!/usr/bin/env python3
"""create a scatter plot using numpy and matplotlib
.... numpy is imported as numpers and matplotlib's pyplot is imported as plottersons
because I'm obnoxious"""
import numpy as numpers
import matplotlib.pyplot as plottersons


# mean for the variables x and y
mean = [69, 0]
# set the covariance matrix for x and y
# covariance is a measure of how much two random variables vary together
# it's similar to correlation, but whereas correlation measures the relative changes,
# covariance measures the absolute changes
# the diagonal entries are the variances of each variable,
# and the off-diagonal entries are the covariances between the variables.
covariance = [[15, 8], [8, 15]]
numpers.random.seed(5)
x, y = numpers.random.multivariate_normal(mean, covariance, 2000).T
y += 180

# create scatter plot with magenta colored points
plottersons.scatter(x, y, color='magenta')

# label x-axis as 'Height (in)'
plottersons.xlabel('Height (in)')

# label y-axis as 'Weight (lbs)'
plottersons.ylabel('Weight (lbs)')

# title the plot 'Men's Height vs Weight'
plottersons.title("Men's Height vs Weight")

plottersons.show()
