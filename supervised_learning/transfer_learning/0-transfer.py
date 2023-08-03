#!/usr/bin/env python3
"""
Otherwise known as megaNet
megaNet: A Deep Learning Model for CIFAR-10 Classification

This script implements a deep learning model for classifying images in the CIFAR-10 dataset.
    The model is built using the Keras library with TensorFlow backend.

The model architecture is based on the MobileNetV2 architecture, pre-trained on the ImageNet dataset.
    The top layers of the MobileNetV2 model are replaced with custom dense layers to adapt the model to the CIFAR-10 classification task.
    The model is first trained with the base MobileNetV2 layers frozen, and then fine-tuned with some of the base layers unfrozen for better adaptation to the CIFAR-10 dataset.

The script also includes data preprocessing and augmentation steps. The CIFAR-10 images are preprocessed according to the requirements of the MobileNetV2 model.
    Data augmentation is performed to increase the diversity of the training data and improve the model's generalization performance.

The model training process includes the use of callbacks for early stopping, model checkpointing, and learning rate scheduling.
    Early stopping is used to prevent overfitting by stopping the training process when the validation performance stops improving.
    Model checkpointing is used to save the model weights at the end of each epoch if the model's performance on the validation set has improved.
    Learning rate scheduling is used to reduce the learning rate periodically for better convergence of the model.

The script includes a main function that loads the CIFAR-10 data, preprocesses the data, defines the model architecture,
    compiles the model, and trains the model. The main function also includes the steps for fine-tuning the model after the initial training.

This script is intended to be run from the command line.

Functions:
    preprocess_data(X, Y): Preprocesses the CIFAR-10 data.

Classes:
    LearningRateSchedulerCallback: A Keras Callback for learning rate scheduling.

Notes:
    The script requires TensorFlow and Keras to be installed. It also requires access to the pre-trained MobileNetV2 weights,
        which are automatically downloaded by the Keras library if not already present.
"""
import tensorflow.keras as K


class LearningRateSchedulerCallback(K.callbacks.Callback):
    """
    Callback to adjust the learning rate according to a schedule.
    This class inherits from the Keras Callback class.
    """
    def on_epoch_end(self, epoch, logs=None):
        """
        This method is called at the end of an epoch.
        If the epoch number is a multiple of 5, it reduces the learning rate by a factor of 10.
        """
        if (epoch+1) % 5 == 0:
            lr = K.backend.get_value(self.model.optimizer.lr)
            K.backend.set_value(self.model.optimizer.lr, lr * 0.1)
            print(" ...Adjusted learning rate to:", lr*0.1)

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
        featurewise_center=False,  # Set input mean to 0 over the dataset
        featurewise_std_normalization=False,  # Divide inputs by std of the dataset
        rotation_range=10,  # Degree range for random rotations
        width_shift_range=0.1,  # Range for random horizontal shifts
        height_shift_range=0.1,  # Range for random vertical shifts
        horizontal_flip=True,  # Randomly flip inputs horizontally
        zoom_range=0.2,  # Range for random zoom
        brightness_range=[0.8, 1.2])  # Range for picking a brightness shift value
    datagen.fit(x_train)

    # Load the base model: MobileNetV2 with pre-trained ImageNet weights, excluding the top dense layers
    base_model = K.applications.MobileNetV2(weights='imagenet', include_top=False)

    # Define the model structure
    model = K.models.Sequential([
        # Use a lambda layer to resize images according to model's expected input shape
        K.layers.Lambda(lambda image: K.backend.resize_images(image, 4, 4, "channels_last"), input_shape=(32, 32, 3)),

        # Add the base model
        base_model,

        # Flatten the output of the base model to 1 dimension
        K.layers.Flatten(),

        # Add a dense layer with 1024 units and ReLU activation
        K.layers.Dense(1024, activation='relu', kernel_regularizer=K.regularizers.l2(0.01)),  # Add L2 regularization

        # Add batch normalization
        K.layers.BatchNormalization(),

        # Add dropout
        K.layers.Dropout(0.5),

        # Add a final dense layer with 10 units (for the 10 classes) with softmax activation
        #   to output probabilities for the classes
        K.layers.Dense(10, activation='softmax')
    ])

    # Compile the model with chosen optimizer
    model.compile(optimizer=K.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define early stopping
    early_stopping = K.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    # Define model checkpointing
    model_checkpoint = K.callbacks.ModelCheckpoint('cifar10.h5', save_best_only=True)

    # Create a learning rate scheduler callback
    lr_scheduler = LearningRateSchedulerCallback()

    # Train the model with early stopping, model checkpointing, and data augmentation
    model.fit(datagen.flow(x_train, y_train, batch_size=64),
              steps_per_epoch=len(x_train) / 64, epochs=20,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping, model_checkpoint, lr_scheduler])

    # Load the pre-trained model
    model = K.models.load_model('cifar10.h5')

    # Unfreeze layers
    for layer in model.layers[-20:]:
        layer.trainable = True

    # Compile the model with a lower learning rate
    model.compile(optimizer=K.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define early stopping
    early_stopping = K.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    # Define model checkpointing for the fine-tuned model
    model_checkpoint = K.callbacks.ModelCheckpoint('cifar10_finetuned.h5', save_best_only=True)

    # Create a learning rate scheduler callback
    lr_scheduler = LearningRateSchedulerCallback()

    # Fine-tune the model with early stopping, model checkpointing, learning rate scheduling and data augmentation
    model.fit(datagen.flow(x_train, y_train, batch_size=64),
              steps_per_epoch=len(x_train) / 64, epochs=24,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping, model_checkpoint, lr_scheduler])
