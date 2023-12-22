#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

autoencoder = __import__('2-convolutional').autoencoder

encoder, decoder, auto = autoencoder((28, 28, 1), [16, 8, 8], (4, 4, 8))
if len(auto.layers) == 3:
    print(auto.layers[0].input_shape == [(None, 28, 28, 1)])
    print(auto.layers[1] is encoder)
    print(auto.layers[2] is decoder)

with open('1-test', 'w+') as f:
    f.write(auto.loss + '\n')
    f.write(auto.optimizer.__class__.__name__ + '\n')

with open('2-test', 'w+') as f:
    try:
        f.write(encoder.layers[0].__class__.__name__ + '\n')
        f.write(str(encoder.layers[0].input_shape) + '\n')
    except:
        f.write('FAIL\n')
    for layer in encoder.layers[1:]:
        try:
            f.write(layer.__class__.__name__ + '\n')
            if layer.__class__.__name__ is 'Conv2D':
                f.write(layer.activation.__name__ + '\n')
            f.write(str(layer.input_shape) + '\n')
            f.write(str(layer.output_shape) + '\n')
        except:
            f.write('FAIL\n')

with open('3-test', 'w+') as f:
    try:
        f.write(decoder.layers[0].__class__.__name__ + '\n')
        f.write(str(decoder.layers[0].input_shape) + '\n')
    except:
        f.write('FAIL\n')
    for layer in decoder.layers[1:]:
        try:
            f.write(layer.__class__.__name__ + '\n')
            if layer.__class__.__name__ is 'Conv2D':
                f.write(layer.activation.__name__ + '\n')
            f.write(str(layer.input_shape) + '\n')
            f.write(str(layer.output_shape) + '\n')
        except:
            f.write('FAIL\n')
