#!/usr/bin/env python3
"""
update train model to also save the best iteration of the model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    network: model to train
    data: numpy.ndarray of shape (m, nx) containing the input data
    labels: one-hot numpy.ndarray of shape
        (m, classes) containing the labels of data
    batch_size: size of the batch used for mini-batch gradient descent
    epochs: the number of passes through data for mini-batch gradient descent
    validation_data: data to validate the model with, if not None
    early_stopping: boolean indicates whether early stopping should be used
        early stopping should only be performed if validation_data exists
        early stopping should be based on validation loss
    patience: the patience used for early stopping
    learning_rate_decay: boolean that indicates whether
        learning rate decay should be used
        ...learning rate decay only performed if validation_data exists
        ...decay performed using inverse time decay
        ...learning rate decays in a stepwise fashion after each epoch
        ...each time the learning rate updates, Keras prints a message
    alpha: the initial learning rate
    decay_rate: the decay rate
    save_best: boolean indicating whether to save the model
        after each epoch if it is the best
        ...a model is considered the best
           if its validation loss is the lowest that the model has obtained
    filepath: the file path where the model should be saved
    verbose: boolean that determines
    ...if output should be printed during training
    shuffle: boolean that determines
        whether to shuffle the batches every epoch
        ...normally, it is a good idea to shuffle, but for reproducibility,
            we have chosen to set the default to False.
    Returns the History object generated after training the model
    """
    callbacks = []

    if validation_data:
        if early_stopping:
            early_stopping = K.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience
                )
            callbacks.append(early_stopping)

        if learning_rate_decay:
            def scheduler(epoch):
                return alpha / (1 + decay_rate * epoch)
            lr_decay = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
            callbacks.append(lr_decay)

        if save_best:
            checkpoint = K.callbacks.ModelCheckpoint(
                filepath=filepath, monitor='val_loss', verbose=1,
                save_best_only=save_best, mode='min'
                )
            callbacks.append(checkpoint)

    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)
    return history
