import tensorflow as tf
import numpy as np
import mnist_input
import multi_mnist_cnn
from matplotlib import pyplot as plt
from sinkhorn import gumbel_sinkhorn, sinkhorn_operator

import util
import random

tf.set_random_seed(94305)
random.seed(94305)

flags = tf.app.flags
flags.DEFINE_integer('M', 1, 'batch size')
flags.DEFINE_integer('n', 3, 'number of elements to compare at a time')
flags.DEFINE_integer('l', 5, 'number of digits')
flags.DEFINE_float('tau', 5, 'temperature (dependent meaning)')
flags.DEFINE_string('method', 'deterministic_neuralsort',
                    'which method to use?')
flags.DEFINE_integer('n_s', 5, 'number of samples')
flags.DEFINE_integer('num_epochs', 200, 'number of epochs to train')
flags.DEFINE_float('lr', 1e-4, 'initial learning rate')

FLAGS = flags.FLAGS

n_s = FLAGS.n_s
NUM_EPOCHS = FLAGS.num_epochs
M = FLAGS.M
n = FLAGS.n
l = FLAGS.l
tau = FLAGS.tau
method = FLAGS.method
initial_rate = FLAGS.lr

train_iterator, val_iterator, test_iterator = mnist_input.get_iterators(
    l, n, 10 ** l - 1, minibatch_size=M)

false_tensor = tf.convert_to_tensor(False)
evaluation = tf.placeholder_with_default(false_tensor, ())
temp = tf.cond(evaluation,
               false_fn=lambda: tf.convert_to_tensor(tau, dtype=tf.float32),
               true_fn=lambda: tf.convert_to_tensor(1e-10, dtype=tf.float32)
               )

experiment_id = 'median-%s-M%d-n%d-l%d-t%d' % (method, M, n, l, tau * 10)
checkpoint_path = 'checkpoints/%s/' % experiment_id

handle = tf.placeholder(tf.string, ())
X_iterator = tf.data.Iterator.from_string_handle(
    handle,
    (tf.float32, tf.float32, tf.float32, tf.float32),
    ((M, n, l * 28, 28), (M,), (M, n), (M, n))
)

X, y, median_scores, true_scores = X_iterator.get_next()

true_scores = tf.expand_dims(true_scores, 2)
P_true = util.neuralsort(true_scores, 1e-10)
n_prime = n


def get_median_probs(P):
    median_strip = P[:, n // 2, :]
    median_total = tf.reduce_sum(median_strip, axis=1, keepdims=True)
    probs = median_strip / median_total
    # print(probs)
    return probs


if method == 'vanilla':
    with tf.variable_scope("phi"):
        representations = multi_mnist_cnn.deepnn(l, X, 10)
    representations = tf.reshape(representations, [M, n * 10])
    fc1 = tf.layers.dense(representations, 10, tf.nn.relu)
    fc2 = tf.layers.dense(fc1, 10, tf.nn.relu)
    fc3 = tf.layers.dense(fc2, 10, tf.nn.relu)
    y_hat = tf.layers.dense(fc3, 1)
    y_hat = tf.squeeze(y_hat)
    loss_phi = tf.reduce_sum(tf.squared_difference(y_hat, y))
    loss_theta = loss_phi
    prob_median_eval = 0

elif method == 'sinkhorn':
    with tf.variable_scope('phi'):
        representations = multi_mnist_cnn.deepnn(l, X, n)
        pre_sinkhorn = tf.reshape(representations, [M, n, n])
    with tf.variable_scope('theta'):
        regression_candidates = multi_mnist_cnn.deepnn(l, X, 1)
        regression_candidates = tf.reshape(
            regression_candidates, [M, n])

    P_hat = sinkhorn_operator(pre_sinkhorn, temp=temp)
    prob_median = get_median_probs(P_hat)

    point_estimates = tf.reduce_sum(
        prob_median * regression_candidates, axis=1)
    exp_loss = tf.squared_difference(y, point_estimates)

    loss_phi = tf.reduce_mean(exp_loss)
    loss_theta = loss_phi

    P_hat_eval = sinkhorn_operator(pre_sinkhorn, temp=1e-20)
    prob_median_eval = get_median_probs(P_hat_eval)

elif method == 'gumbel_sinkhorn':
    with tf.variable_scope('phi'):
        representations = multi_mnist_cnn.deepnn(l, X, n)
        pre_sinkhorn_orig = tf.reshape(representations, [M, n, n])
        pre_sinkhorn = tf.tile(pre_sinkhorn_orig, [
                               n_s, 1, 1])
        pre_sinkhorn += util.sample_gumbel([n_s * M, n, n])

    with tf.variable_scope('theta'):
        regression_candidates = multi_mnist_cnn.deepnn(l, X, 1)
        regression_candidates = tf.reshape(
            regression_candidates, [M, n])

    P_hat = sinkhorn_operator(pre_sinkhorn, temp=temp)
    prob_median = get_median_probs(P_hat)
    prob_median = tf.reshape(prob_median, [n_s, M, n])

    point_estimates = tf.reduce_sum(
        prob_median * regression_candidates, axis=2)
    exp_loss = tf.squared_difference(y, point_estimates)

    loss_phi = tf.reduce_mean(exp_loss)
    loss_theta = loss_phi

    P_hat_eval = sinkhorn_operator(pre_sinkhorn_orig, temp=1e-20)
    prob_median_eval = get_median_probs(P_hat_eval)

elif method == 'deterministic_neuralsort':
    with tf.variable_scope('phi'):
        scores = multi_mnist_cnn.deepnn(l, X, 1)
        scores = tf.reshape(scores, [M, n, 1])

    P_hat = util.neuralsort(scores, temp)
    P_hat_eval = util.neuralsort(scores, 1e-20)

    with tf.variable_scope('theta'):
        regression_candidates = multi_mnist_cnn.deepnn(l, X, 1)
        regression_candidates = tf.reshape(
            regression_candidates, [M, n])

    losses = tf.squared_difference(
        regression_candidates, tf.expand_dims(y, 1))
    prob_median = get_median_probs(P_hat)
    prob_median_eval = get_median_probs(P_hat_eval)

    point_estimates = tf.reduce_sum(
        prob_median * regression_candidates, axis=1)
    exp_loss = tf.squared_difference(y, point_estimates)

    point_estimates_eval = tf.reduce_sum(
        prob_median_eval * regression_candidates, axis=1)
    exp_loss_eval = tf.squared_difference(y, point_estimates)

    loss_phi = tf.reduce_mean(exp_loss)
    loss_theta = tf.reduce_mean(exp_loss_eval)

elif method == 'stochastic_neuralsort':
    with tf.variable_scope('phi'):
        scores = multi_mnist_cnn.deepnn(l, X, 1)
        scores = tf.reshape(scores, [M, n, 1])
        scores = tf.tile(scores, [n_s, 1, 1])
        scores += util.sample_gumbel([M * n_s, n, 1])

    P_hat = util.neuralsort(scores, temp)
    P_hat_eval = util.neuralsort(scores, 1e-20)

    with tf.variable_scope('theta'):
        regression_candidates = multi_mnist_cnn.deepnn(l, X, 1)
        regression_candidates = tf.reshape(
            regression_candidates, [M, n])

    res_y = tf.expand_dims(y, 1)

    losses = tf.squared_difference(regression_candidates, res_y)

    prob_median = get_median_probs(P_hat)
    prob_median = tf.reshape(prob_median, [n_s, M, n])
    prob_median_eval = get_median_probs(P_hat_eval)
    prob_median_eval = tf.reshape(prob_median_eval, [n_s, M, n])

    exp_losses = tf.reduce_sum(prob_median * losses, axis=2)
    exp_losses_eval = tf.reduce_sum(
        prob_median_eval * losses, axis=2)

    loss_phi = tf.reduce_mean(exp_losses)
    loss_theta = tf.reduce_mean(exp_losses_eval)
else:
    raise ValueError("No such method.")

num_losses = M * n_s if method == 'stochastic_neuralsort' or method == 'gumbel_sinkhorn' else M

correctly_identified = tf.reduce_sum(
    prob_median_eval * median_scores) / num_losses

phi = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='phi')
theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='theta')

train_phi = tf.train.AdamOptimizer(
    initial_rate).minimize(loss_phi, var_list=phi)

if method != 'vanilla':
    train_theta = tf.train.AdamOptimizer(initial_rate).minimize(
        loss_phi, var_list=theta)
    train_step = tf.group(train_phi, train_theta)
else:
    train_step = train_phi

saver = tf.train.Saver()

sess = tf.Session()
logfile = open('./logs/%s.log' % experiment_id, 'w')


def prnt(*args):
    print(*args)
    print(*args, file=logfile)


sess.run(tf.global_variables_initializer())
train_sh, validate_sh, test_sh = sess.run([
    train_iterator.string_handle(),
    val_iterator.string_handle(),
    test_iterator.string_handle()
])

TRAIN_PER_EPOCH = mnist_input.TRAIN_SET_SIZE // (l * M)
VAL_PER_EPOCH = mnist_input.VAL_SET_SIZE // (l * M)
TEST_PER_EPOCH = mnist_input.TEST_SET_SIZE // (l * M)
best_val = float('inf')
tiebreaker_val = -1


def save_model(epoch):
    saver.save(sess, checkpoint_path + 'checkpoint', global_step=epoch)


def load_model():
    filename = tf.train.latest_checkpoint(checkpoint_path)
    if filename == None:
        raise Exception("No model found.")
    print("Loaded model %s." % filename)
    saver.restore(sess, filename)


def train(epoch):
    loss_train = []
    for _ in range(TRAIN_PER_EPOCH):
        _, l = sess.run([train_step, loss_phi],
                        feed_dict={handle: train_sh})
        loss_train.append(l)
    prnt('Average loss:', sum(loss_train) / len(loss_train))


def test(epoch, val=False):
    global best_correct_val
    c_is = []
    l_vs = []
    for _ in range(VAL_PER_EPOCH if val else TEST_PER_EPOCH):
        c_i, l_v = sess.run([correctly_identified, loss_phi], feed_dict={
                            handle: validate_sh if val else test_sh, evaluation: True})
        c_is.append(c_i)
        l_vs.append(l_v)

    c_i = sum(c_is) / len(c_is)
    l_v = sum(l_vs) / len(l_vs)

    if val:
        prnt("Validation set: correctly identified %f, mean squared error %f" %
             (c_i, l_v))
        if l_v < best_val:
            best_val = l_v
            prnt('Saving...')
            save_model(epoch)
    else:
        prnt("Test set: correctly identified %f, mean squared error %f" %
             (c_i, l_v))


for epoch in range(1, NUM_EPOCHS + 1):
    prnt('Epoch', epoch, '(%s)' % experiment_id)
    train(epoch)
    test(epoch, val=True)
    logfile.flush()
load_model()
test(epoch, val=False)

sess.close()
logfile.close()
