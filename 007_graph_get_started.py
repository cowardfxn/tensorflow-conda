#!/bin/python3
# encoding: utf-8

import tensorflow as tf
import numpy as np


class MnistData():
    def __init__(self, size=100):
        self.size = size
        self.index = 0
        mnist = tf.keras.datasets.mnist
        (self.train_x, self.train_y), (self.val_x, self.val_y) = mnist.load_data()
        self.train_x = self.train_x.reshape(-1, 28 * 28).astype(np.float32)
        self.val_x = self.val_x.reshape(-1, 28 * 28).astype(np.float32)

    def next_batch(self):
        datas = self.train_x[self.index: self.index + self.size]
        labels = self.train_y[self.index: self.index + self.size]
        self.index += self.size
        return datas, labels

    def get_validation(self):
        return self.val_x, self.val_y


x = tf.placeholder('float', [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder('float', [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correction_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correction_prediction, 'float'))

mnist_data = MnistData(100)
val_x, val_y = mnist_data.get_validation()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    with tf.device('/gpu:0'):  # will use gpu by default if gpu available
        # TODO Is this initializing right?
        sess.run(init)

        # change label to one-hot
        val_y = sess.run(tf.one_hot(val_y, depth=10))
        acc = 0
        for i in range(int(1e3)):
            batch_xs, batch_ys = mnist_data.next_batch()
            # change label to one-hot
            batch_ys = sess.run(tf.one_hot(batch_ys, depth=10))
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            if i % 100 == 0:
                acc = sess.run(accuracy, feed_dict={x: val_x, y_: val_y})
                print(acc)

        print("Final accuracy: {}".format(acc))