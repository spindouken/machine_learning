#!/usr/bin/env python3
import tensorflow.keras as K


class LearningRateSchedulerCallback(K.callbacks.Callback):
    """Learning rate scheduler callback."""
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % 5 == 0:
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

    # Load the base model: MobileNetV2 with pre-trained ImageNet weights, excluding the top dense layers
    base_model = K.applications.MobileNetV2(weights='imagenet', include_top=False)

    # Define the model structure
    model = K.models.Sequential([
        # Use a lambda layer to resize images from 32x32 to 128x128 to match the input size 
        #   that MobileNetV2 was trained on
        K.layers.Lambda(lambda image: K.backend.resize_images(image, 4, 4, "channels_last"), input_shape=(32, 32, 3)),

        # Add the base model (MobileNetV2)
        base_model,

        # Flatten the output of the base model to 1 dimension
        K.layers.Flatten(),

        # Add a dense layer with 1024 units and ReLU activation
        K.layers.Dense(1024, activation='relu'),

        # Add dropout
        K.layers.Dropout(0.5),

        # Add a final dense layer with 10 units (for the 10 classes) with softmax activation 
        #   to output probabilities for the classes
        K.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=K.optimizers.RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define early stopping
    early_stopping = K.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    # Define model checkpointing
    model_checkpoint = K.callbacks.ModelCheckpoint('cifar10_lilTinyNet.h5', save_best_only=True)

    # Create a learning rate scheduler callback
    lr_scheduler = LearningRateSchedulerCallback()

    # Train the model with early stopping and model checkpointing
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stopping, model_checkpoint, lr_scheduler])
