#!/usr/bin/env python3
"""numPY and matplotlib present plotty plottersons
draw standard line graph that plots the cubes of an array
"""
import numpy as numpers
import matplotlib.pyplot


# creates an array with the cubes of the numbers from 0 to 10
y = numpers.arange(0, 11) ** 3

# create an array for the x-axis values
x = numpers.arange(0, 11)

# plot y as a solid red line
matplotlib.pyplot.plot(x, y, 'r')

# set x-axis range from 0 to 10
matplotlib.pyplot.xlim(0, 10)

# display plot
matplotlib.pyplot.show()
