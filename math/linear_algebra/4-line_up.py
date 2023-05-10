#!/usr/bin/env python3
"""Adds two arrays and returns the product"""


def add_arrays(arr1, arr2):
    """takes two input arrays (lists of integers/floats)
    and adds them element-wise.
    If the array have different lenghts,
    the function returns None,
    as element-wise addtion only works
    with arrays of equal length"""
    if len(arr1) != len(arr2):
        return None
    return [arr1[index] + arr2[index] for index in range(len(arr1))]
