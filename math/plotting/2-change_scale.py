#!/usr/bin/env python3
"""create logarithmicly scaled line graph using numpy and matplotlib"""
import numpy as numpers
import matplotlib.pyplot as plottersons


x = numpers.arange(0, 28651, 5730)
r = numpers.log(0.5)
t = 5730
y = numpers.exp((r / t) * x)

# matplotlib line graph
plottersons.plot(x, y)

plottersons.xlabel('Time (years)')

plottersons.ylabel('Fraction Remaining')

plottersons.title('Exponential Decay of C-14')

# apply log scale to y-axis
plottersons.yscale('log')

# set x-axis range from 0 to 28650
plottersons.xlim(0, 28650)

plottersons.show()
