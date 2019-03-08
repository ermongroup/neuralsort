import numpy as np
import random
import tensorflow as tf
import math
from statistics import median
from tensorflow.examples.tutorials.mnist import input_data

TRAIN_SET_SIZE = 55000
VAL_SET_SIZE = 5000
TEST_SET_SIZE = 10000


def select_digit(split, d):
    return split.images[np.nonzero(split.labels[:, d])]


def split_digits(split):
    return [select_digit(split, d) for d in range(10)]


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_digits = split_digits(mnist.train)
validation_digits = split_digits(mnist.validation)
test_digits = split_digits(mnist.test)


def get_multi_mnist_input(l, n, low, high, digset=train_digits):
    multi_mnist_sequences = []
    values = []
    for i in range(n):
        mnist_digits = []
        num = random.randint(low, high)
        values.append(num)

        for i in range(l):
            digit = num % 10
            num //= 10
            ref = digset[digit]
            mnist_digits.insert(0, ref[np.random.randint(0, ref.shape[0])])
        multi_mnist_sequence = np.concatenate(mnist_digits)
        multi_mnist_sequence = np.reshape(multi_mnist_sequence, (-1, 28))
        multi_mnist_sequences.append(multi_mnist_sequence)
    multi_mnist_batch = np.stack(multi_mnist_sequences)
    vals = np.array(values)
    med = int(median(values))
    arg_med = np.equal(vals, med).astype('float32')
    arg_med /= np.sum(arg_med)
    return multi_mnist_batch, med, arg_med, vals


def get_iterator(l, n, window_size, digset, minibatch_size=None):
    low, high = 0, 10 ** l - 1

    def input_generator():
        while True:
            window_begin = random.randint(low, high - window_size)
            ret = get_multi_mnist_input(
                l, n, window_begin, window_begin + window_size, digset)
            yield ret
    mm_data = tf.data.Dataset.from_generator(
        input_generator,
        (tf.float32, tf.float32, tf.float32, tf.float32),
        ((n, l * 28, 28), (), (n,), (n,))
    )
    if minibatch_size:
        mm_data = mm_data.batch(minibatch_size)
    mm_data = mm_data.prefetch(10)
    return mm_data.make_one_shot_iterator()


def get_iterators(l, n, window_size, minibatch_size=None, val_repeat=None):
    return get_iterator(l, n, window_size, train_digits, minibatch_size=minibatch_size), \
        get_iterator(l, n, window_size, validation_digits, minibatch_size=minibatch_size), \
        get_iterator(l, n, window_size, test_digits,
                     minibatch_size=minibatch_size)


def test_iterators():
    a, b, c = get_iterators(5, 10, 100)
    with tf.Session() as sess:
        for d in [a, b, c]:
            print(sess.run(d.get_next()))
