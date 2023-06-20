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
        train_op = tf.get_collection("train_op")[0]

        m = X_train.shape[0]

        total_batches = m // batch_size
        if m % batch_size != 0:
            total_batches += 1

        for epoch in range(epochs + 1):
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_accuracy = sess.run(
                accuracy, feed_dict={x: X_train, y: Y_train}
                )
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(
                accuracy, feed_dict={x: X_valid, y: Y_valid}
                )
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
                for batch in range(total_batches):
                    start = batch * batch_size
                    end = start + batch_size
                    X_batch = X_shuffled[start:end]
                    Y_batch = Y_shuffled[start:end]
                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                    if batch != 0 and batch % 100 == 0: # print every 100 batches
                        cost = sess.run(
                            loss, feed_dict={x: X_batch, y: Y_batch}
                            )
                        acc = sess.run(
                            accuracy, feed_dict={x: X_batch, y: Y_batch}
                            )
                        print("\tStep {}:".format(batch))
                        print("\t\tCost: {}".format(cost))
                        print("\t\tAccuracy: {}".format(acc))

        save_path = saver.save(sess, save_path)
    return save_path
