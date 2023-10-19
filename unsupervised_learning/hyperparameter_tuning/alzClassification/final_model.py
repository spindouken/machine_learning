#!/usr/bin/env python3
import pickle
import numpy as np
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from load_and_preprocess import load_data


def train_final_model(best_params):
    """
    Train the final model using the best hyperparameters obtained from Bayesian optimization.

    Parameters:
    best_params (numpy.ndarray): NumPy array containing the best hyperparameters obtained from Bayesian optimization.
    """
    print("Training final model with best hyperparameters...")
    
    # Input validation and type checking
    if not isinstance(best_params, np.ndarray):
        raise ValueError("best_params must be a NumPy array.")
        
    if best_params.size < 5:
        raise ValueError("Insufficient number of hyperparameters in best_params.")
        
    # Extract all best hyperparameters from best_params
    learning_rate = best_params[0]
    dense_units = int(best_params[1])
    dropout_rate = best_params[2]
    l2_weight = best_params[3]
    batch_size = int(best_params[4])
    
    # Load training and validation datasets
    trainingGenerator, validationGenerator = load_data('data/')
    
    # Create and compile model with best hyperparameters
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Flatten(),
        Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_weight)),
        Dropout(dropout_rate),
        Dense(int(dense_units / 2), activation='relu', kernel_regularizer=l2(l2_weight)),
        Dropout(dropout_rate),
        Dense(4, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Checkpoint to save the best model
    checkpoint = ModelCheckpoint("final_model.h5", monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min')
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    # Train the model
    model.fit(trainingGenerator, epochs=10, validation_data=validationGenerator,
              batch_size=batch_size, callbacks=[checkpoint, early_stopping])

    print("Final model trained and saved as 'final_model.h5'.")

# Check if the best_params.pkl file exists
if not os.path.exists("best_params.pkl"):
    raise FileNotFoundError("best_params.pkl not found.")

# Load best_params.pkl
with open("best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

# Input validation for best_params
if best_params is None or not isinstance(best_params, np.ndarray):
    raise ValueError("Invalid or empty best_params.")

train_final_model(best_params)
