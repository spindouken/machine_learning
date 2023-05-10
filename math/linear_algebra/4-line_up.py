#!/usr/bin/env python3
def add_arrays(arr1, arr2):
    if len(arr1) != len(arr2):
        return None
    return [arr1[index] + arr2[index] for index in range(len(arr1))]
