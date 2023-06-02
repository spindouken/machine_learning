#!/usr/bin/env python3
"""placeholder"""
import numpy as np

class Neuron:
    """placeholder"""
    def __init__(self, nx):
        """placeholder"""
        if not isinstance(nx, int):
            raise TypeError ("nx must be a positive integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
