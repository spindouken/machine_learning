#!/usr/bin/env python3
"""create a stacked bar chart using numpy and matplotlib"""
import numpy as numpers
import matplotlib.pyplot as plottersons


numpers.random.seed(5)
fruit = numpers.random.randint(0, 20, (4,3))

# names
people = ['Farrah', 'Fred', 'Felicia']

# names and colors for each type of fruit
fruits = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

# create figure
figure, axis = plottersons.subplots()

# create stacked bar chart
for i in range(len(fruits)):
    axis.bar(people, fruit[i], width=0.5, label=fruits[i],
             color=colors[i], bottom=numpers.sum(fruit[:i], axis=0))

axis.set_ylabel('Quantity of Fruit')
axis.set_title('Number of Fruit per Person')
axis.set_yticks(numpers.arange(0, 81, 10))

axis.legend()

plottersons.show()
