#!/usr/bin/env python3
"""
Train mini batch
"""
import tensorflow as tf
import numpy as np


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    train neural network model using mini-batch gradient descent
    X_train: numpy.ndarray of shape (m, 784) with the training data
        m: number of data points
        784: number of input features
    Y_train: one-hot numpy.ndarray of shape (m, 10) with the training labels
        10: number of classes the model should classify
    X_valid: numpy.ndarray of shape (m, 784) with the validation data
    Y_valid: one-hot numpy.ndarray of shape (m, 10) with the validation labels
    batch_size: number of data points in a batch
    epochs: number of times the training should pass through the whole dataset
    load_path: path from which to load the model
    save_path: path to where the model should be saved after training
    Returns the path where the model was saved
    """
    shuffle_data = __import__('2-shuffle_data').shuffle_data

    with tf.Session() as sess:
        # 1) import meta graph and restore session
        saver = tf.train.import_meta_graph("{}.meta".format(load_path))
        saver.restore(sess, load_path)

        # 2) Get the following tensors and ops from the collection
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        # calculate number of iterations per epoch
        m = X_train.shape[0]

        # calculate and print cost and accuracy before the first epoch
        train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
        train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
        valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
        valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        print("After 0 epochs:")
        print("\tTraining Cost: {}".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_accuracy))
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_accuracy))

        # 3) loop over epochs
        for epoch in range(epochs):
            # shuffle data
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
            # loop over the batches
            for i in range(0, m, batch_size):
                # extract batches
                X_batch = X_shuffled[i:i + batch_size]
                Y_batch = Y_shuffled[i:i + batch_size]
                # train model
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                # print step cost and accuracy after every 100 steps
                if (i // batch_size) % 100 == 0:
                    cost = sess.run(loss, feed_dict={x: X_batch, y: Y_batch})
                    acc = sess.run(accuracy,
                                   feed_dict={x: X_batch, y: Y_batch})
                    print("\tStep {}:".format(i // batch_size))
                    print("\t\tCost: {}".format(cost))
                    print("\t\tAccuracy: {}".format(acc))

            # print training and validation cost after each epoch
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_accuracy = sess.run(
                accuracy, feed_dict={x: X_train, y: Y_train}
                )
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(
                accuracy, feed_dict={x: X_valid, y: Y_valid}
                )
            print("After {} epochs:".format(epoch + 1))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
        # 4) save session
        save_path = saver.save(sess, save_path)
    return save_path
