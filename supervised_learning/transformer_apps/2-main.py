#!/usr/bin/env python3

Dataset = __import__("2-dataset").Dataset
import tensorflow as tf

data = Dataset()
print("got here")
for pt, en in data.data_train.take(1):
    print(pt, en)
for pt, en in data.data_valid.take(1):
    print(pt, en)
