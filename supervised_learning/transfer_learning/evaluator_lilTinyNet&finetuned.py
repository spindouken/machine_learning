#!/usr/bin/env python3

import tensorflow.keras as K
preprocess_data = __import__('lilTinyNet').preprocess_data

# to fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase 

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10_lilTinyNet.h5')
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)


import tensorflow.keras as K
preprocess_data = __import__('lilTinyNet_finetuner').preprocess_data

K.learning_phase = K.backend.learning_phase 

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10_finetuned_lilTinyNet.h5')
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)
