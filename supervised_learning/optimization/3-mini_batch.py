#!/usr/bin/env python3
"""Train mini batch"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train,
                     X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """checker bug test"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(load_path))
        saver.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")

        m = X_train.shape[0]

        train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
        train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
        valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
        valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        print("After 0 epochs:")
        print("\tTraining Cost: {}".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_accuracy))
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_accuracy))

        for epoch in range(epochs):
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
            steps = m // batch_size
            if steps * batch_size < m:
                steps += 1
            for i in range(steps):
                start = i * batch_size
                end = i * batch_size + batch_size
                if end > m:
                    end = m
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                if i != 0 and (i + 1) % 100 == 0:
                    cost = sess.run(loss, feed_dict={x: X_batch, y: Y_batch})
                    acc = sess.run(accuracy,
                                   feed_dict={x: X_batch, y: Y_batch})
                    print("\tStep {}:".format(i + 1))
                    print("\t\tCost: {}".format(cost))
                    print("\t\tAccuracy: {}".format(acc))

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
        save_path = saver.save(sess, save_path)
    return save_path
