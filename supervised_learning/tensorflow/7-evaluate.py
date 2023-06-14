#!/usr/bin/env python3
"""evaluation function of tensorflow network output"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    loads tensorflow network from a save path,
        then evaluates performance on provided input data and labels
    X: numpy.ndarray - input data to evaluate
    Y: numpy.ndarray - one-hot labels for X
    save_path: str - the location to load the model from

    Returns:
    Tuple containing the network's prediction, accuracy, and loss, respectively
    """
    # create tensorflow session
    with tf.Session() as sess:
        # load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        # get relevant tensors from the graph's collections
        x = tf.get_collection('x')[0]  # input data
        y = tf.get_collection('y')[0]  # labels
        y_pred = tf.get_collection('y_pred')[0]  # predictions
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        # evaluate network on provided data
        prediction, network_accuracy, cost = sess.run([y_pred, accuracy, loss],
                                                      feed_dict={x: X, y: Y})

    return prediction, network_accuracy, cost
