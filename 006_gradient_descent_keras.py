#!/bin/python3
# encoding: utf-8

import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

X = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
y = tf.constant([[10], [20]], dtype=tf.float32)


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units=1, kernel_initializer=tf.zeros_initializer(),
                                           bias_initializer=tf.zeros_initializer())

    def call(self, inputs):
        return self.dense(inputs)


model = Linear()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
for _ in range(10000):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

print([e.numpy() for e in model.variables])
