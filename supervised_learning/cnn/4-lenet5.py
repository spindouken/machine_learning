#!/usr/bin/env python3
"""
builds a modified version of the LeNet-5 architecture using tensorflow
"""
import tensorflow as tf


def lenet5(x, y):
    """
    x is a tf.placeholder of shape (m, 28, 28, 1)
        ...containing the input images for the network
        m: the number of images
    y is a tf.placeholder of shape (m, 10)
        ...containing the one-hot labels for the network

    The model consists of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes
        All layers requiring initialization,
            ...initialize their kernels with the he_normal initialization
            ...method: tf.contrib.layers.variance_scaling_initializer()
        All hidden layers requiring activation
            ...use the relu activation function
    Returns:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization
            ...(with default hyperparameters)
        a tensor for the loss of the netowrk
        a tensor for the accuracy of the network
    """
    he_normal_initializer = tf.contrib.layers.variance_scaling_initializer()

    # first convolutional layer
    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = tf.layers.Conv2D(filters=6,
                             kernel_size=(5, 5),
                             padding='same',
                             activation='relu',
                             kernel_initializer=he_normal_initializer)(x)
    # first pooling layer
    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv1)
    # second convolutional layer
    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = tf.layers.Conv2D(filters=16,
                             kernel_size=(5, 5),
                             padding='valid',
                             activation='relu',
                             kernel_initializer=he_normal_initializer)(pool1)
    # second pooling layer
    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv2)
    # flattened output from second pooling layer
    flattened = tf.layers.Flatten()(pool2)
    # fully connected layers
    # first fully connected layer
    # Fully connected layer with 120 nodes
    fc1 = tf.layers.Dense(units=120,
                          kernel_initializer=he_normal_initializer,
                          activation='relu')(flattened)
    # second fully connected layer
    # fully connected layer with 84 nodes
    fc2 = tf.layers.Dense(units=84,
                          kernel_initializer=he_normal_initializer,
                          activation='relu')(fc1)
    # third fully connected layer
    # fully connected softmax output layer with 10 nodes
    softmax = tf.layers.Dense(units=10, activation=tf.nn.softmax,
                          kernel_initializer=he_normal_initializer)(fc2)

    # get softmax cross entropy loss
    crossEntropy_loss = tf.losses.softmax_cross_entropy(y, fc3)

    # training op that utilizes Adam optimzation
    adamOptimizer = tf.train.AdamOptimizer().minimize(
        crossEntropy_loss
        )

    correctPredictions = tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1))

    # create tensor for the accuracy of the network
    accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32))

    return softmax_output, adamOptimizer, crossEntropy_loss, accuracy
