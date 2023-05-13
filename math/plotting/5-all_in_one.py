#!/usr/bin/env python3
"""display charts from problems 1-4 in one display"""
import numpy as numpers
import matplotlib.pyplot as plottersons


# set seed for reproducibility
numpers.random.seed(5)

y0 = numpers.arange(0, 11) ** 3
mean = [69, 0]
cov = [[15, 8], [8, 15]]
x1, y1 = numpers.random.multivariate_normal(mean, cov, 2000).T
y1 += 180
x2 = numpers.arange(0, 28651, 5730)
r2 = numpers.log(0.5)
t2 = 5730
y2 = numpers.exp((r2 / t2) * x2)
x3 = numpers.arange(0, 21000, 1000)
r3 = numpers.log(0.5)
t31 = 5730
t32 = 1600
y31 = numpers.exp((r3 / t31) * x3)
y32 = numpers.exp((r3 / t32) * x3)
student_grades = numpers.random.normal(68, 15, 50)

# create figure and grid gridSpec
figure = plottersons.figure(layout="constrained")
gridSpec = figure.add_gridspec(3, 2)

# first subplot
exponential = figure.add_subplot(gridSpec[0, 0])
exponential.plot(y0, color='red')
exponential.set_xlim([0, 10])

# second subplot
scatterplot = figure.add_subplot(gridSpec[0, 1])
scatterplot.scatter(x1, y1, color='magenta', marker='.')
scatterplot.set_xlabel('Height (in)', fontsize='x-small')
scatterplot.set_ylabel('Weight (lbs)', fontsize='x-small')
scatterplot.set_title("Men's Height vs Weight", fontsize='x-small')

# third subplot
decayPlot = figure.add_subplot(gridSpec[1, 0])
decayPlot.semilogy(x2, y2)
decayPlot.set_xlim(0, 28650)
decayPlot.set_xlabel("Time (years)", fontsize='x-small')
decayPlot.set_ylabel("Fraction Remaining", fontsize='x-small')
decayPlot.set_title("Exponential Decay of C-14", fontsize='x-small')

# fourth subplot
radioactiveDecayPlot = figure.add_subplot(gridSpec[1, 1])
radioactiveDecayPlot.plot(x3, y31, c="red", linestyle="--", label="C-14")
radioactiveDecayPlot.plot(x3, y32, c="green", label="Ra-226")
radioactiveDecayPlot.set_xlim(0, 20000)
radioactiveDecayPlot.set_ylim(0, 1)
radioactiveDecayPlot.set_title("Exponential Decay of Radioactive Elements", fontsize='x-small')
radioactiveDecayPlot.set_xlabel("Time (years)", fontsize='x-small')
radioactiveDecayPlot.set_ylabel("Fraction Remaining", fontsize='x-small')
radioactiveDecayPlot.legend()

# fifth subplot
gradesHistogram = figure.add_subplot(gridSpec[2, :])
gradesHistogram.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
gradesHistogram.set_xlim(0, 100)
gradesHistogram.set_xticks(ticks=range(0, 101, 10))
gradesHistogram.set_ylim(0, 30)
gradesHistogram.set_title("Project A", fontsize='x-small')
gradesHistogram.set_ylabel("Number of Students", fontsize='x-small')
gradesHistogram.set_xlabel("Grades", fontsize='x-small')

figure.suptitle("All in One")
plottersons.show()
