#!/usr/bin/env python3
"""create line graph w/ standard exponential curves using numpy and matplotlib
This code in particular is expressing the exponentrial decay
...of some particular radioactive elements
the graph includes different visual lines and a legend
"""
import numpy as numpers
import matplotlib.pyplot as plotterssons


x = numpers.arange(0, 21000, 1000)
r = numpers.log(0.5)
t1 = 5730
t2 = 1600
y1 = numpers.exp((r / t1) * x)
y2 = numpers.exp((r / t2) * x)

# plot y1 as a dashed red line
plotterssons.plot(x, y1, 'r--', label='C-14')

# plot y2 as a solid green line
plotterssons.plot(x, y2, 'g-', label='Ra-226')

plotterssons.xlabel('Time (years)')

plotterssons.ylabel('Fraction Remaining')

plotterssons.title('Exponential Decay of Radioactive Elements')

# set x-axis range from 0 to 20000
plotterssons.xlim(0, 20000)

# set y-axis range from 0 to 1
plotterssons.ylim(0, 1)

# add a legend in the upper right hand corner of the plot
plotterssons.legend(loc='upper right')

plotterssons.show()
