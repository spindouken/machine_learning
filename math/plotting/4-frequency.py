#!/usr/bin/env python3
"""create a histogram using numpy and matplotlib"""
import numpy as numpers
import matplotlib.pyplot as plotters


# set seed to ensure the same sequence of 
# 'random' numbers are generated on each run
numpers.random.seed(5)
# generates a normalling distributed set of 50 grades
# with a mean of 68 and standard deviation of 15
student_grades = numpers.random.normal(68, 15, 50)

# create histogram containing student scores in a certain range
plotters.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

plotters.xlabel('Grades')

plotters.ylabel('Number of Students')

plotters.title('Project A')

plotters.ylim(0, 30)

plotters.show()
