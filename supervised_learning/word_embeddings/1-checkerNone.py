#!/usr/bin/env python3

tf_idf = __import__("1-tf_idf").tf_idf

sentences = [
    "Holberton school is Awesome!",
    "Machine learning is awesome",
    "NLP is the future!",
    "The children are our future",
    "Our children's children are our grandchildren",
    "The cake was not very good",
    "No one said that the cake was not very good",
    "Life is beautiful",
]
E, F = tf_idf(sentences)
print(E)
print(F)
