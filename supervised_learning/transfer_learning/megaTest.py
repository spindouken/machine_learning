#!/usr/bin/env python3
"""
evaluator and visualizer for lilTinyNet and finetuned lilTinyNet
includes model architecture visualization and model evaluation on CIFAR-10 dataset
"""

import tensorflow.keras as K
from tensorflow.keras.utils import plot_model
import numpy as np
preprocess_data = __import__('lilTinyNet').preprocess_data

# to fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase 

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10_lilTinyNet.h5')

# Display the model architecture
model.summary()

# Evaluate the model
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)

# Make predictions
predictions = model.predict(X_p)

# Display the first 5 predictions
print("First 5 predictions: ", np.argmax(predictions, axis=1)[:5])

# Visualize the model architecture
plot_model(model, to_file='model.png', show_shapes=True)

# Repeat the same for the finetuned model
preprocess_data = __import__('lilTinyNet_finetuner').preprocess_data

K.learning_phase = K.backend.learning_phase 

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10_finetuned_lilTinyNet.h5')

# Display the model architecture
model.summary()

# Evaluate the model
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)

# Make predictions
predictions = model.predict(X_p)

# Display the first 5 predictions
print("First 5 predictions: ", np.argmax(predictions, axis=1)[:5])

# Visualize the model architecture
plot_model(model, to_file='finetuned_model.png', show_shapes=True)
