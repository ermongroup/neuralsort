import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# M: minibatch size
# n: number of items in each sequence
# s: scores

np.set_printoptions(precision=4, suppress=True)
eps = 1e-20

def bl_matmul(A, B):
  return tf.einsum('mij,jk->mik', A, B)

def br_matmul(A, B):
  return tf.einsum('ij,mjk->mik', A, B)

# s: M x n x 1
# neuralsort(s): M x n x n
def neuralsort(s, tau = 1):
  A_s = s - tf.transpose(s, perm=[0, 2, 1])
  A_s = tf.abs(A_s) 
  # As_ij = |s_i - s_j|

  n = tf.shape(s)[1] 
  one = tf.ones((n, 1), dtype = tf.float32)

  B = bl_matmul(A_s, one @ tf.transpose(one))
  # B_:k = (A_s)(one)

  K = tf.range(n) + 1
  # K_k = k

  C = bl_matmul(
    s, tf.expand_dims(tf.cast(n + 1 - 2 * K, dtype = tf.float32), 0)
  )
  # C_:k = (n + 1 - 2k)s

  P = tf.transpose(C - B, perm=[0, 2, 1])
  # P_k: = (n + 1 - 2k)s - (A_s)(one)
 
  P = tf.nn.softmax(P / tau, -1)
  # P_k: = softmax( ((n + 1 - 2k)s - (A_s)(one)) / tau )

  return P

# Pi: M x n x n row-stochastic
def prop_any_correct (P1, P2):
  z1 = tf.argmax(P1, axis=-1) 
  z2 = tf.argmax(P2, axis=-1)
  eq = tf.equal(z1, z2)
  eq = tf.cast(eq, dtype=tf.float32)
  correct = tf.reduce_mean(eq, axis=-1)
  return tf.reduce_mean(correct)

# Pi: M x n x n row-stochastic
def prop_correct (P1, P2):
  z1 = tf.argmax(P1, axis=-1) 
  z2 = tf.argmax(P2, axis=-1)
  eq = tf.equal(z1, z2)
  correct = tf.reduce_all(eq, axis=-1)
  return tf.reduce_mean(tf.cast(correct, tf.float32))

def sample_gumbel(shape, eps = 1e-20): 
	U = tf.random_uniform(shape, minval=0, maxval=1)
	return -tf.log(-tf.log(U + eps) + eps)

# s: M x n
# P: M x n x n
# returns: M
def pl_log_density(log_s, P):
  log_s = tf.expand_dims(log_s, 2) # M x n x 1
  ordered_log_s = P @ log_s # M x n x 1
  ordered_log_s = tf.squeeze(ordered_log_s, squeeze_dims=[-1]) # M x n
  potentials = tf.exp(ordered_log_s)
  n = log_s.get_shape().as_list()[1]
  max_log_s = [
    tf.reduce_max(ordered_log_s[:, k:], axis=1, keepdims=True)
    for k in range(n)
  ] # [M x 1] x n
  adj_log_s = [
    ordered_log_s - max_log_s[k]
    for k in range(n)
  ] # [M x n] x n
  potentials = [
    tf.exp(adj_log_s[k][:, k:])
    for k in range(n)
  ] # [M x n] x n
  denominators = [
    tf.reduce_sum(potentials[k], axis=1, keepdims=True)
    for k in range(n)
  ] # [M x 1] x n
  log_denominators = [
    tf.squeeze(tf.log(denominators[k]) + max_log_s[k], squeeze_dims=[1])
    for k in range(n)
  ] # [M] x n
  log_denominator = tf.add_n(log_denominators) # M
  log_potentials = ordered_log_s # M x n x 1
  log_potential = tf.reduce_sum(log_potentials, 1) # M
  log_likelihood = log_potential - log_denominator
  return log_likelihood
