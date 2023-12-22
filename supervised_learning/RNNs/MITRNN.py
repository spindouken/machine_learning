#!/usr/bin/env python3
"""
orrrr... use:
tf.keras.layers.SimpleRNN(rnn_units)
"""
import tensorflow as tf


Class MyRNNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, input_dim, output_dim):
        super(MyRNNCell, self).__init__()

        # initialize weight matrices
        self.W_xh = self.add_weight([rnn_units, input_dim])
        self.W_hh = self.add_weight([rnn_units, rnn_units])
        self.W_hy = self.add_weight([output_dim, rnn_units])

        # initialize hidden state to zeros
        self.h = tf.zeros([rnn_units, 1])
    
    def call(self, x):
        # update hidden state
        self.h = tf.math.tanh(self.W_xh * self.h + self.W_xh * x)

        # compute output
        output = self.W_hy * self.h

        # return current output and hidden state
        return output, self.h
