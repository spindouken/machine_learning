#!/usr/bin/env python3
"""
Perform bayesian optimization on a neural network
    The neural network to be optimized is in the function create_model
    The hyperparameter space to be searched is defined in the function define_hyperparameter_space
    The dataset can be found in the data directory
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import GPyOpt
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
from tensorflow.keras import backend as K
import os
import pickle
import datetime
from tensorflow.keras.applications import DenseNet169
import matplotlib.pyplot as plt
import numpy as np

iterationCount = 0 
bestValidationLoss = float('inf')
bestF1Score = -1.0
bestHyperparameters = None
bestModel = None


def load_data(data_dir, batch_size=32):
    """
    load the Alzheimer MRI dataset from a specified directory

    args:
        data_dir (str): directory where the dataset is stored
        batch_size (int): batch size for the data generator
            will default to 32 if not specified when calling the function

    Returns:
        trainingGenerator, validationGenerator: data generators for training and validation sets
    """
    datagen = ImageDataGenerator(validation_split=0.2)  # 20% data for validation

    trainingGenerator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    validationGenerator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return trainingGenerator, validationGenerator


def create_model(input_shape, num_classes):
    """
    Create the custom CNN model for Alzheimer's classification

    args:
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of classes in the dataset.
    """
    # pre-trained DenseNet169 as the base model
    base_model = DenseNet169(
        include_top=False, weights="imagenet", input_shape=input_shape
    )

    # freeze all layers except the last two dense blocks
    for layer in base_model.layers[:-17]:
        layer.trainable = False

    # get the output tensor of the base model
    base_model_output = base_model.output

    x = GlobalAveragePooling2D()(base_model_output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(num_classes, activation="softmax")(x)

    # compile Model
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def define_hyperparameter_space():
    """
    Define the hyperparameter space for Bayesian optimization.
    
    Returns:
        domain (list): List of dictionaries specifying the hyperparameter space.
    """
    domain = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
        {'name': 'dense_units', 'type': 'discrete', 'domain': (128, 256, 512)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.3, 0.7)},
        {'name': 'l2_weight', 'type': 'continuous', 'domain': (1e-5, 1e-3)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64)}
    ]
    return domain


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
    trainingGenerator, validationGenerator = load_data('alzClassification/data/', batch_size=batch_size)

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


def save_and_plot(optimizer):
    """
    save photos of and print evaluations and plot convergence
    """
    # Plot and save convergence
    optimizer.plot_convergence()
    plt.savefig('convergence.png')

    # Plot acquisition function
    optimizer.plot_acquisition()
    plt.savefig('acquisition_function.png')

    # Save optimization evaluations to a text file
    with open('bayes_opt_MRIalz.txt', 'w') as f:
        f.write(str(optimizer.get_evaluations()))
        best_params_so_far = optimizer.X[np.argmin(optimizer.Y)]
        f.write(f"\nBest parameters so far: {best_params_so_far}\n")
        f.write(f"Mean objective: {np.mean(optimizer.Y)}\n")
        f.write(f"Standard Deviation: {np.std(optimizer.Y)}\n")


def main():
    """
    Run Bayesian optimization to tune hyperparameters using GPyOpt
        and save the best hyperparameters (in best_params.pkl) to be used in training final_model.py
        .pkl file will come with timestamp to account for multiple bayesian optimization runs

    Main function utilizes the following functions to perform Bayesian optimization:
        define_hyperparameter_space.py
        BayesianOptimization.py
        load_and_preprocess.py
        save_and_plot.py

    Note: bayesian optimization is actually performed in BayesianOptimization.py
    """
    # create a directory for best models if it doesn't exist
    if not os.path.exists('best_models'):
        os.makedirs('best_models')

    print("Starting Bayesian optimization...")

    # use function (from main folder)
    #   which defined the hyperparameter space for Bayesian optimization
    domainExpansion = define_hyperparameter_space()

    # initialize bayesian optimization
    # add initial_design_numdata=0 to avoid random initialization
    #   and speed up optimization (for bug testing)
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=BayesianOptimization,
        domain=domainExpansion,
        acquisition_type="EI",  # expected improvement
        exact_feval=True,
        maximize=False,
    )

    # specify max run count for optimization
    optimizer.run_optimization(max_iter=1)

    best_model = get_best_model()
    bestHyperparameters = get_best_hyperparameters()

    if best_model is not None:
        best_model.save(f"best_models/bestModel_{bestHyperparameters}.h5")

    print(
        "Bayesian optimization completed. Next step: Use the best hyperparameters to train your final model."
    )

    timestamp = datetime.datetime.now().strftime("%m-%d-%y-%H:%M")
    filename = f"best_params_{timestamp}.pkl"
    # retrieve best parameters from optimizer and save them to be used in final_model.py
    best_params = optimizer.x_opt
    with open(filename, "wb") as f:
        pickle.dump(best_params, f)

    # save and plot the results of the optimization
    #   this will save the convergence plot as 'convergence.png'
    #   and the optimization evaluations as 'bayes_opt_MRIalz.txt'
    save_and_plot(optimizer)
    print(
        "Best hyperparameters saved to best_params_{timestamp}.pkl. Convergence and acquisition visualizations were stored in their respective .png files."
    )

if __name__ == "__main__":
    main()
