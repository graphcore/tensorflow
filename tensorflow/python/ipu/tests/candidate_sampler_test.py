# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from functools import partial
from tensorflow.python.ipu.config import IPUConfig
from collections import Counter
import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.training import gradient_descent
from tensorflow.python.platform import googletest


def softmax_cifar(sampled=True, k=25, iters=1000):
  # Perform the softmax on k out of 100 classes when training on CIFAR-100
  with ops.Graph().as_default():
    N = 100
    BATCH_SIZE = 8
    HIDDEN_DIM = 3 * 32**2
    (x_train,
     y_train), _ = keras.datasets.cifar100.load_data(label_mode="fine")
    x_train = np.reshape(x_train, [x_train.shape[0], -1])
    x_train = x_train.astype('float32') / 255
    y_train = np.reshape(y_train.astype(np.int32), [-1, 1])
    dataset = dataset_ops.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    infeed = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def body(X, Y):
      # Y is [BATCH_SIZE, 1]
      # X is [BATCH_SIZE, HIDDEN_DIM]
      weights = variable_scope.get_variable("weights", [N, HIDDEN_DIM],
                                            dtypes.float32,
                                            init_ops.ones_initializer)
      biases = variable_scope.get_variable("biases", [N], dtypes.float32,
                                           init_ops.ones_initializer)
      if sampled:
        # Sample k classes instead, calculate the softmax on them and update
        # only their weights
        loss = nn_impl.sampled_softmax_loss(weights=weights,
                                            biases=biases,
                                            labels=Y,
                                            inputs=X,
                                            num_sampled=k,
                                            num_classes=N,
                                            num_true=1,
                                            remove_accidental_hits=False)
      else:
        # The last dense layer outputting class logits
        logits = math_ops.matmul(X, array_ops.transpose(weights))
        logits = nn.bias_add(logits, biases)
        # Softmax to turn logits into probabilities on all N classes
        labels_one_hot = array_ops.one_hot(Y, N)
        loss = nn.softmax_cross_entropy_with_logits(labels=labels_one_hot,
                                                    logits=logits)

      enqueue_op = outfeed.enqueue(math_ops.reduce_mean(loss))
      train_op = gradient_descent.GradientDescentOptimizer(0.1).minimize(loss)
      return train_op, enqueue_op

    def my_net():
      return ipu.loops.repeat(iters, body, [], infeed)

    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      run_op = ipu.ipu_compiler.compile(my_net)

    with session_lib.Session() as sess:
      sess.run(infeed.initializer)
      sess.run(variables.global_variables_initializer())
      sess.run(run_op)
      losses = sess.run(outfeed.dequeue())
  return losses


def from_same_distribution(samples1, samples2, percent_error=1):
  # We could use the Kolmogorov-Smirnov test here to check the two sets of
  # samples come from the same distribution, but it would need a scipy
  # import. Instead, we make sure both devices output similar numbers of
  # samples for each class s.t. the mean difference is < 1% of the total
  # number of requested samples
  freq1 = Counter(samples1)
  freq2 = Counter(samples2)
  freq1.subtract(freq2)
  mean_diff = np.mean(np.abs(list(freq1.values()))) / samples1.size * 100
  return mean_diff < percent_error


def generate_ops(sess,
                 unique=False,
                 dist="uniform",
                 true_classes=None,
                 num_samples=1000,
                 num_classes=50,
                 num_iters=1,
                 seed=None):
  if dist == "uniform":
    sampling_op = candidate_sampling_ops.uniform_candidate_sampler
  elif dist == "log_uniform":
    sampling_op = candidate_sampling_ops.log_uniform_candidate_sampler
  else:
    raise Exception(f"generate_op: dist {dist} not supported.")

  def body(x):
    return [
        sampling_op(true_classes=x,
                    num_true=true_classes.shape[1],
                    num_sampled=num_samples,
                    unique=unique,
                    range_max=num_classes,
                    seed=seed)
    ] * num_iters

  with ops.device('cpu'):
    inp = array_ops.placeholder(np.int64, true_classes.shape)

  with ipu.scopes.ipu_scope("/device:IPU:0"):
    ipu_op = ipu.ipu_compiler.compile(body, inputs=[inp])

  cfg = IPUConfig()
  tu.add_hw_ci_connection_options(cfg)
  cfg.configure_ipu_system()

  _maybe_unpack = lambda x: x if num_iters > 1 else x[0]
  run_ipu = lambda: _maybe_unpack(
      sess.run(ipu_op, feed_dict={inp: true_classes}))
  run_cpu = lambda: _maybe_unpack(
      sess.run(body(inp), feed_dict={inp: true_classes}))
  return run_ipu, run_cpu


class TestCandidateSampler(test_util.TensorFlowTestCase):
  UNIQUE_TEST_CASES = [[1, 1000], [100, 1000], [900, 1000], [1000, 1000]]
  TEST_CASES = UNIQUE_TEST_CASES + [[1000, 100]]

  def verify(self,
             calc_expected_fn,
             k,
             N,
             unique=False,
             dist='uniform',
             rtol=1e-06,
             atol=1e-06):
    """
    Run a series of tests to verify a specific candidate sampling op given some
    parameters k and N. The tests are:
    1. It compiles
    2. All samples are in [0, N)
    3. The calculated expectations for the entire range [0, N) are similar
       enough to the CPU calculated expectations.
    4. The calculated expectations for the entire range [0, N) are similar
       enough to what we expect for the op (via calc_expected_fn).

    Additional tests for the unique case:
    5. All samples are unique
    6. All expectations are in [0, 1]

    Additional tests for the uniform case:
    7. The standard deviation of the expectations is close to 0 (a.k.a. they are
       all the same)
    """
    with session_lib.Session() as sess:
      np.random.seed(42)
      ipu.utils.reset_ipu_seed(42)
      random_seed.set_random_seed(42)

      true_classes = np.arange(N).reshape([1, -1])

      run_ipu, run_cpu = generate_ops(sess,
                                      true_classes=true_classes,
                                      num_samples=k,
                                      num_classes=N,
                                      unique=unique,
                                      dist=dist)
      results = run_ipu()
      samples, true_expected, samples_expected = results

      # All samples should be in the range [0, N)
      self.assertTrue(((samples >= 0) & (samples < N)).all())

      if unique:
        # Unique expectations should all be in [0, 1]
        self.assertTrue(((true_expected >= 0) & (true_expected <= 1)).all())
        self.assertTrue(
            ((samples_expected >= 0) & (samples_expected <= 1)).all())

        # Unique samples shouldn't contain any duplicates
        self.assertTrue(len(set(samples)) == len(samples))

      if dist == "uniform":
        self.assertAlmostEqual(np.std(true_expected), 0, places=5)

      # Ensure IPU expectation is what we expect
      # Note that the current implementation of both the IPU and native CPU
      # versions of the ops use a heuristic for expectation based on the
      # number of tries it took to sample all k samples.
      self.assertAllClose(true_expected,
                          calc_expected_fn(true_classes),
                          atol=0.1,
                          rtol=0.005)

      # Compare with CPU samples
      cpu_results = run_cpu()
      _, cpu_true_expected, _ = cpu_results
      self.assertAllClose(true_expected,
                          cpu_true_expected,
                          atol=atol,
                          rtol=rtol)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testUniformSamplerWithReplacement(self):
    """
    Test the uniform candidate sampling op over a range of (k, N) test
    cases.
    """
    # Expectation for uniform is always k/p
    for k, N in self.TEST_CASES:

      def func(k, N, x):
        return np.ones_like(x) * (k / N)

      self.verify(partial(func, k, N),
                  k,
                  N,
                  dist="uniform",
                  unique=False,
                  atol=1e-06)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testUniformSamplerWithoutReplacement(self):
    """
    Test the unique uniform candidate sampling op over a range of (k, N) test
    cases.
    """
    for k, N in self.UNIQUE_TEST_CASES:
      # Expectation for unique uniform is still always k/p
      def func(k, N, x):
        return np.ones_like(x) * (k / N)

      self.verify(partial(func, k, N),
                  k,
                  N,
                  dist="uniform",
                  unique=True,
                  atol=0.05)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testLogUniformSamplerWithReplacement(self):
    """
    Test the log uniform candidate sampling op over a range of (k, N) test
    cases.
    """
    for k, N in self.TEST_CASES:
      # Expectation for log-uniform is k * the pdf
      def log_uniform_k_pdf(k, N, x):
        return k * np.log((x + 2) / (x + 1)) / np.log(N)

      self.verify(partial(log_uniform_k_pdf, k, N),
                  k,
                  N,
                  dist="log_uniform",
                  unique=False,
                  rtol=0.1)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testLogUniformSamplerWithoutReplacement(self):
    """
    Test the unique log uniform candidate sampling op over a range of (k, N)
    test cases. We estimate expected expectation by simulating it on the host.
    """
    NUM_TRIALS = 10000
    for k, N in self.UNIQUE_TEST_CASES:
      # Expectation for unique log-uniform can only be estimated empirically
      X = np.arange(0, N)
      pdf = np.log((X + 2) / (X + 1)) / np.log(N)
      pdf /= np.sum(pdf)
      freqs = np.zeros(N)
      for _ in range(NUM_TRIALS):
        freqs[np.random.choice(X, k, replace=False, p=pdf)] += 1

      def func(freqs, x):
        return (freqs / NUM_TRIALS)[x]

      self.verify(partial(func, freqs),
                  k,
                  N,
                  dist="log_uniform",
                  unique=True,
                  atol=0.1)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testFixedSeed(self):
    """
    If we fix the seed of the candidate sampling ops, they should return the
    same sequence of samples every time the graph is called.
    """
    k = 10
    N = 1000
    true_classes = np.arange(N).reshape([1, -1])
    run_ipu, _ = generate_ops(session_lib.Session(),
                              unique=False,
                              dist="uniform",
                              true_classes=true_classes,
                              num_samples=k,
                              num_classes=N,
                              seed=42)

    # Two separate runs should return the same samples with a fixed seed
    samples, _, _ = run_ipu()
    samples2, _, _ = run_ipu()
    self.assertAllEqual(samples, samples2)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testFixedSeedMultipleIters(self):
    """
    If we fix the seed of the candidate sampling ops, they should return
    different samples each time they're called in the same graph, but the same
    sequence of samples every time the graph is called.
    """
    k = 10
    N = 1000
    np.random.seed(42)

    # Run the op 5 times inside the graph - each iter should return different
    # samples
    true_classes = np.arange(N).reshape([1, -1])
    run_ipu, _ = generate_ops(session_lib.Session(),
                              unique=True,
                              dist="uniform",
                              true_classes=true_classes,
                              num_samples=k,
                              num_classes=N,
                              num_iters=5,
                              seed=42)

    # The per-iteration samples may be different, but the same iteration from
    # different session calls should be the same
    all_samples = [x[0] for x in run_ipu()]
    all_samples2 = [x[0] for x in run_ipu()]
    for s1, s2 in zip(all_samples, all_samples2):
      self.assertAllEqual(s1, s2)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testSampledSoftmaxCompiles(self):
    """
    Make sure that sampled softmax compiles
    """
    with ops.Graph().as_default():
      softmax_cifar(sampled=True, k=25)


if __name__ == "__main__":
  googletest.main()
