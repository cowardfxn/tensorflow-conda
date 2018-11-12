#!/bin/python3
# encoding: utf-8

import tensorflow as tf
import numpy as np
import PIL.Image
# from cStringIO import StringIO
from IPython.display import clear_output, Image, display
import scipy.ndimage as ndi


def display_fractal(a: np.ndarray, fmt='jpeg'):
    a_cyclic = (6.28 * a / 40).reshape(list(a.shape) + [1])
    img = np.concatenate([
        20 + 20 * np.cos(a_cyclic),
        20 + 220 * np.sin(a_cyclic),
        155 - 80 * np.cos(a_cyclic)
    ], 2)
    img[a == a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))  # limit the value in an array
    f = './out_img'
    PIL.Image.fromarray(a).save(f, fmt)

    # terminal or jupyter display
    # with open(f, 'rb') as ifs:
    #     display(Image(data=ifs.read()),)


Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Z = X + 1j * Y

xs = tf.constant(Z.astype('complex64'))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, 'float32'))

init = tf.global_variables_initializer()
zs_ = zs * zs + xs
not_diverged = tf.abs(zs_) < 4
step = tf.group(
    zs.assign(zs_),
    ns.assign_add(tf.cast(not_diverged, 'float32'))
)

with tf.Session() as sess:
    sess.run(init)

    for i in range(200):
        sess.run(step)

    display_fractal(sess.run(ns))
