#!/usr/bin/env python3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from load_and_preprocess import load_data
from create_model import create_model
from tensorflow.keras.metrics import AUC
from tensorflow.keras import backend as K

iterationCount = 0 
bestValidationLoss = float('inf')
bestF1Score = -1.0
bestHyperparameters = None
bestModel = None


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

def get_best_model():
    global bestModel
    return bestModel

def get_best_hyperparameters():
    global bestHyperparameters
    return bestHyperparameters

def BayesianOptimization(params):
    """
    Creates model from create_model with training and validation data,
        performs bayesian optimization on the model with the given hyperparameter space,
        and returns the satisfacing metric.
    
    Parameters:
        params (dict): Hyperparameters to optimize.

    Function is set to only save the best_model after training is complete
    """
    global bestValidationLoss,bestHyperparameters, bestModel, bestF1Score, iterationCount

    # extract hyperparameters from params
    params = params[0]
    learning_rate = params[0]
    dense_units = int(params[1])
    dropout_rate = params[2]
    l2_weight = params[3]
    batch_size = int(params[4])

    hyperparameters = f"lr={learning_rate:.5f}, du={dense_units}, dr={dropout_rate:.3f}, l2={l2_weight:.5f}, bs={batch_size}"

    iterationCount += 1
    print(f"Iteration: {iterationCount}")
    print(f"Optimizing with: {hyperparameters}")

    # load training data
    trainingGenerator, validationGenerator = load_data('data', batch_size=batch_size)

    # create model from create_model.py
    model = create_model((128, 128, 3), 4)  # Assuming 128x128 images and 4 classes

    # compile model with new hyperparameters
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', AUC(name='auc'), f1_score])

    # early_stopping will stop training if F1 score doesn't improve for # epochs (patience)
    early_stopping = EarlyStopping(monitor='val_f1_score', patience=2, verbose=1, mode='min')

    history = model.fit(trainingGenerator, epochs=1, validation_data=validationGenerator, 
                        batch_size=batch_size, callbacks=[early_stopping])

    # return validation loss from the last epoch
    currentValidationLoss = history.history['val_loss'][-1]
    currentF1Score = history.history['val_f1_score'][-1]

    # check if the current model performs better than the best model stored in memory
    if currentF1Score > bestF1Score:
        bestF1Score = currentF1Score
        bestHyperparameters = hyperparameters
        bestModel = model

    # update best validation loss to be printed during training
    if currentValidationLoss < bestValidationLoss:
        bestValidationLoss = currentValidationLoss

    print(f"Validation loss with these parameters: {currentValidationLoss}")
    print("AUC-ROC:", history.history['val_auc'][-1])
    print("F1-Score:", currentF1Score)
    print(f"Current best validation loss: {bestValidationLoss}, with hyperparameters: {bestHyperparameters}")
    print(f"Current best F1 score: {bestF1Score}, with hyperparameters: {bestHyperparameters}")

    return currentF1Score
