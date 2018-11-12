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


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder('float', [None, 28 * 28])
y_ = tf.placeholder('float', [None, 10])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

keep_prop = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prop)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

mnist_data = MnistData(50)
val_x, val_y = mnist_data.get_validation()

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    val_y = sess.run(tf.one_hot(val_y, depth=10))

    for i in range(int(2e4)):
        train_x, train_y = mnist_data.next_batch()
        train_y = sess.run(tf.one_hot(train_y, depth=10))
        sess.run(train_step, feed_dict={x: train_x, y_: train_y, keep_prop: 0.5})
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: train_x, y_: train_y, keep_prop: 1.0})
            print('Step {}, training accuracy: {}'.format(i, train_accuracy))

    print('Test accuracy: {}'.format(sess.run(accuracy, feed_dict={x: val_x, y_: val_y, keep_prop: 1.0})))
