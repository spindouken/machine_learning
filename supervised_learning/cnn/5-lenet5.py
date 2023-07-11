"""
builds a modified version of the LeNet-5 architecture using keras
"""
import tensorflow.keras as K


def lenet5(x):
    """
    X is a K.Input of shape (m, 28, 28, 1)
        ...containing the input images for the network
        m: the number of images
    The model should consist of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes
        All layers requiring initialization, initialize
            ...their kernels with the he_normal initialization method
        All hidden layers requiring activation
            ...use the relu activation function
    Returns: a K.Model compiled to use Adam optimization
        ...(with default hyperparameters) and accuracy metrics
    """
    he_normal_initializer = K.initializers.HeNormal()
    # first convolutional layer
    conv1 = K.layers.Conv2D(filters=6,
                            kernel_size=(5, 5),
                            padding='same',
                            activation='relu',
                            kernel_initializer=he_normal_initializer)(X)
    # first pooling layer
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv1)
    # second convolutional layer
    conv2 = K.layers.Conv2D(filters=16,
                            kernel_size=(5, 5),
                            padding='valid',
                            activation='relu',
                            kernel_initializer=he_normal_initializer)(pool1)
    # second pooling layer
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv2)
    # flattened output from second pooling layer
    flattened = K.layers.Flatten()(pool2)
    # fully connected layers
    # first fully connected layer
    fc1 = K.layers.Dense(units=120,
                         kernel_initializer=he_normal_initializer,
                         activation='relu')(flattened)
    # second fully connected layer
    fc2 = K.layers.Dense(units=84,
                         kernel_initializer=he_normal_initializer,
                         activation='relu')(fc1)
    # output layer (final fully connected layer before softmax)
    fc3 = K.layers.Dense(units=10,
                         kernel_initializer=he_normal_initializer,
                         activation='softmax')(fc2)
    lenet5_model = K.Model(inputs=X, outputs=fc3)
    lenet5_model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return lenet5_model
