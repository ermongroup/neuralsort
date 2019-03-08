# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Parts of the original code have been removed.

import tensorflow as tf

def sinkhorn_operator(log_alpha, n_iters=20, temp=1.0):
  """Performs incomplete Sinkhorn normalization to log_alpha.
  By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
  with positive entries can be turned into a doubly-stochastic matrix
  (i.e. its rows and columns add up to one) via the succesive row and column
  normalization.
  -To ensure positivity, the effective input to sinkhorn has to be
  exp(log_alpha) (elementwise).
  -However, for stability, sinkhorn works in the log-space. It is only at
   return time that entries are exponentiated.
  [1] Sinkhorn, Richard and Knopp, Paul.
  Concerning nonnegative matrices and doubly stochastic
  matrices. Pacific Journal of Mathematics, 1967
  Args:
    log_alpha: 2D tensor (a matrix of shape [N, N])
      or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
    n_iters: number of sinkhorn iterations (in practice, as little as 20
      iterations are needed to achieve decent convergence for N~100)
  Returns:
    A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
      converted to 3D tensors with batch_size equals to 1)
  """

  n = tf.shape(log_alpha)[1]
  log_alpha = tf.reshape(log_alpha, [-1, n, n]) / temp

  for _ in range(n_iters):
    log_alpha -= tf.reshape(tf.reduce_logsumexp(log_alpha, axis=2), [-1, n, 1])
    log_alpha -= tf.reshape(tf.reduce_logsumexp(log_alpha, axis=1), [-1, 1, n])
  return tf.exp(log_alpha)

def gumbel_sinkhorn(log_alpha,
                    temp=1.0, n_samples=1, noise_factor=1.0, n_iters=20,
                    squeeze=True):
  """Random doubly-stochastic matrices via gumbel noise.
  In the zero-temperature limit sinkhorn(log_alpha/temp) approaches
  a permutation matrix. Therefore, for low temperatures this method can be
  seen as an approximate sampling of permutation matrices, where the
  distribution is parameterized by the matrix log_alpha
  The deterministic case (noise_factor=0) is also interesting: it can be
  shown that lim t->0 sinkhorn(log_alpha/t) = M, where M is a
  permutation matrix, the solution of the
  matching problem M=arg max_M sum_i,j log_alpha_i,j M_i,j.
  Therefore, the deterministic limit case of gumbel_sinkhorn can be seen
  as approximate solving of a matching problem, otherwise solved via the
  Hungarian algorithm.
  Warning: the convergence holds true in the limit case n_iters = infty.
  Unfortunately, in practice n_iter is finite which can lead to numerical
  instabilities, mostly if temp is very low. Those manifest as
  pseudo-convergence or some row-columns to fractional entries (e.g.
  a row having two entries with 0.5, instead of a single 1.0)
  To minimize those effects, try increasing n_iter for decreased temp.
  On the other hand, too-low temperature usually lead to high-variance in
  gradients, so better not choose too low temperatures.
  Args:
    log_alpha: 2D tensor (a matrix of shape [N, N])
      or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
    temp: temperature parameter, a float.
    n_samples: number of samples
    noise_factor: scaling factor for the gumbel samples. Mostly to explore
      different degrees of randomness (and the absence of randomness, with
      noise_factor=0)
    n_iters: number of sinkhorn iterations. Should be chosen carefully, in
      inverse corresponde with temp to avoid numerical stabilities.
    squeeze: a boolean, if True and there is a single sample, the output will
      remain being a 3D tensor.
  Returns:
    sink: a 4D tensor of [batch_size, n_samples, N, N] i.e.
      batch_size *n_samples doubly-stochastic matrices. If n_samples = 1 and
      squeeze = True then the output is 3D.
    log_alpha_w_noise: a 4D tensor of [batch_size, n_samples, N, N] of
      noisy samples of log_alpha, divided by the temperature parameter. If
      n_samples = 1 then the output is 3D.
  """
  n = tf.shape(log_alpha)[1]
  log_alpha = tf.reshape(log_alpha, [-1, n, n])
  batch_size = tf.shape(log_alpha)[0]
  log_alpha_w_noise = tf.tile(log_alpha, [n_samples, 1, 1])
  if noise_factor == 0:
    noise = 0.0
  else:
    noise = sample_gumbel([n_samples*batch_size, n, n])*noise_factor
  log_alpha_w_noise += noise
  log_alpha_w_noise /= temp
  sink = sinkhorn_operator(log_alpha_w_noise, n_iters)
  if n_samples > 1 or squeeze is False:
    sink = tf.reshape(sink, [n_samples, batch_size, n, n])
    sink = tf.transpose(sink, [1, 0, 2, 3])
    log_alpha_w_noise = tf.reshape(
        log_alpha_w_noise, [n_samples, batch_size, n, n])
    log_alpha_w_noise = tf.transpose(log_alpha_w_noise, [1, 0, 2, 3])
  return sink, log_alpha_w_noise
