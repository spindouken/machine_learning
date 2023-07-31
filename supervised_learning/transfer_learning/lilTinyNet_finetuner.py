#!/usr/bin/env python3
import tensorflow.keras as K


class LearningRateSchedulerCallback(K.callbacks.Callback):
    """Learning rate scheduler callback."""
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % 4 == 0:
            lr = K.backend.get_value(self.model.optimizer.lr)
            K.backend.set_value(self.model.optimizer.lr, lr * 0.1)
            print("Adjusted learning rate to:", lr*0.1)

def preprocess_data(X, Y):
    """
    Pre-processes the data for your model.
    X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data, where m is the number of data points.
    Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X.
    Returns: X_p, Y_p.
    X_p is a numpy.ndarray containing the preprocessed X.
    Y_p is a numpy.ndarray containing the preprocessed Y.
    """
    X_p = K.applications.mobilenet_v2.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == '__main__':
    # Load the CIFAR10 data
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # Pre-process the data
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Data augmentation
    datagen = K.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
    datagen.fit(x_train)

    # Load the pre-trained model
    model = K.models.load_model('cifar10_run2.h5')

    # Unfreeze layers
    for layer in model.layers[-10:]:
        layer.trainable = True

    # Compile the model with a lower learning rate
    model.compile(optimizer=K.optimizers.RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define early stopping
    early_stopping = K.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    # Define model checkpointing for the fine-tuned model
    model_checkpoint = K.callbacks.ModelCheckpoint('cifar10_finetuned.h5', save_best_only=True)

    # Create a learning rate scheduler callback
    lr_scheduler = LearningRateSchedulerCallback()

    # Fine-tune the model with early stopping, model checkpointing and learning rate scheduling
    model.fit(datagen.flow(x_train, y_train, batch_size=32), 
              steps_per_epoch=len(x_train) / 32, epochs=12, 
              validation_data=(x_test, y_test), 
              callbacks=[early_stopping, model_checkpoint, lr_scheduler])
