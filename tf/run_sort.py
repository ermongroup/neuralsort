import tensorflow as tf
import numpy as np
import mnist_input
import multi_mnist_cnn
from sinkhorn import gumbel_sinkhorn, sinkhorn_operator

import util
import random

tf.set_random_seed(94305)
random.seed(94305)

flags = tf.app.flags
flags.DEFINE_integer('M', 1, 'batch size')
flags.DEFINE_integer('n', 3, 'number of elements to compare at a time')
flags.DEFINE_integer('l', 4, 'number of digits')
flags.DEFINE_integer('tau', 5, 'temperature (dependent meaning)')
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
temperature = tf.cond(evaluation,
                      false_fn=lambda: tf.convert_to_tensor(
                          tau, dtype=tf.float32),
                      true_fn=lambda: tf.convert_to_tensor(
                          1e-10, dtype=tf.float32)  # simulate hard sort
                      )

experiment_id = 'sort-%s-M%d-n%d-l%d-t%d' % (method, M, n, l, tau * 10)
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

if method == 'vanilla':
    representations = multi_mnist_cnn.deepnn(l, X, n)
    concat_reps = tf.reshape(representations, [M, n * n])
    fc1 = tf.layers.dense(concat_reps, n * n)
    fc2 = tf.layers.dense(fc1, n * n)
    P_hat_raw = tf.layers.dense(fc2, n * n)
    P_hat_raw_square = tf.reshape(P_hat_raw, [M, n, n])

    P_hat = tf.nn.softmax(P_hat_raw_square, dim=-1)  # row-stochastic!

    losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=P_true, logits=P_hat_raw_square, dim=2)
    losses = tf.reduce_mean(losses, axis=-1)
    loss = tf.reduce_mean(losses)

if method == 'sinkhorn':
    representations = multi_mnist_cnn.deepnn(l, X, n)
    pre_sinkhorn = tf.reshape(representations, [M, n, n])
    P_hat = sinkhorn_operator(pre_sinkhorn, temp=temperature)
    P_hat_logit = tf.log(P_hat)

    losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=P_true, logits=P_hat_logit, dim=2)
    losses = tf.reduce_mean(losses, axis=-1)
    loss = tf.reduce_mean(losses)

if method == 'gumbel_sinkhorn':
    representations = multi_mnist_cnn.deepnn(l, X, n)
    pre_sinkhorn = tf.reshape(representations, [M, n, n])
    P_hat = sinkhorn_operator(pre_sinkhorn, temp=temperature)

    P_hat_sample, _ = gumbel_sinkhorn(
        pre_sinkhorn, temp=temperature, n_samples=n_s)
    P_hat_sample_logit = tf.log(P_hat_sample)

    P_true_sample = tf.expand_dims(P_true, 1)
    P_true_sample = tf.tile(P_true_sample, [1, n_s, 1, 1])

    losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=P_true_sample, logits=P_hat_sample_logit, dim=3)
    losses = tf.reduce_mean(losses, axis=-1)
    losses = tf.reshape(losses, [-1])
    loss = tf.reduce_mean(losses)

if method == 'deterministic_neuralsort':
    scores = multi_mnist_cnn.deepnn(l, X, 1)
    scores = tf.reshape(scores, [M, n, 1])
    P_hat = util.neuralsort(scores, temperature)

    losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=P_true, logits=tf.log(P_hat + 1e-20), dim=2)
    losses = tf.reduce_mean(losses, axis=-1)
    loss = tf.reduce_mean(losses)

if method == 'stochastic_neuralsort':
    scores = multi_mnist_cnn.deepnn(l, X, 1)
    scores = tf.reshape(scores, [M, n, 1])
    P_hat = util.neuralsort(scores, temperature)

    scores_sample = tf.tile(scores, [n_s, 1, 1])
    scores_sample += util.sample_gumbel([M * n_s, n, 1])
    P_hat_sample = util.neuralsort(
        scores_sample, temperature)

    P_true_sample = tf.tile(P_true, [n_s, 1, 1])
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=P_true_sample, logits=tf.log(P_hat_sample + 1e-20), dim=2)
    losses = tf.reduce_mean(losses, axis=-1)
    loss = tf.reduce_mean(losses)
else:
    raise ValueError("No such method.")


def vec_gradient(l):  # l is a scalar
    gradient = tf.gradients(l, tf.trainable_variables())
    vec_grads = [tf.reshape(grad, [-1]) for grad in gradient]  # flatten
    z = tf.concat(vec_grads, 0)  # n_params
    return z


prop_correct = util.prop_correct(P_true, P_hat)
prop_any_correct = util.prop_any_correct(P_true, P_hat)

opt = tf.train.AdamOptimizer(initial_rate)
train_step = opt.minimize(loss)
saver = tf.train.Saver()

# MAIN BEGINS

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
best_correct_val = 0


def save_model(epoch):
    saver.save(sess, checkpoint_path + 'checkpoint', global_step=epoch)


def load_model():
    filename = tf.train.latest_checkpoint(checkpoint_path)
    if filename == None:
        raise Exception("No model found.")
    prnt("Loaded model %s." % filename)
    saver.restore(sess, filename)


def train(epoch):
    loss_train = []
    for _ in range(TRAIN_PER_EPOCH):
        _, l = sess.run([train_step, loss],
                        feed_dict={handle: train_sh})
        loss_train.append(l)
    prnt('Average loss:', sum(loss_train) / len(loss_train))


def test(epoch, val=False):
    global best_correct_val
    p_cs = []
    p_acs = []
    for _ in range(VAL_PER_EPOCH if val else TEST_PER_EPOCH):
        p_c, p_ac = sess.run([prop_correct, prop_any_correct], feed_dict={
                             handle: validate_sh if val else test_sh,
                             evaluation: True})
        p_cs.append(p_c)
        p_acs.append(p_ac)

    p_c = sum(p_cs) / len(p_cs)
    p_ac = sum(p_acs) / len(p_acs)

    if val:
        prnt("Validation set: prop. all correct %f, prop. any correct %f" %
             (p_c, p_ac))
        if p_c > best_correct_val:
            best_correct_val = p_c
            prnt('Saving...')
            save_model(epoch)
    else:
        prnt("Test set: prop. all correct %f, prop. any correct %f" % (p_c, p_ac))


for epoch in range(1, NUM_EPOCHS + 1):
    prnt('Epoch', epoch, '(%s)' % experiment_id)
    train(epoch)
    test(epoch, val=True)
    logfile.flush()
load_model()
test(epoch, val=False)

sess.close()
logfile.close()
