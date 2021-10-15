# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
import itertools

import numpy as np
from absl.testing import parameterized

from tensorflow.python.client import session as sl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python import ipu
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu

# Error threshold for forward pass test.
THRESHOLD = 0.1

# Dimensions of the random data tensor.
DIMS = (64, 64, 4)

# Initialise with a random seed.
SEED = np.random.randint(np.iinfo(np.int32).max, size=[2], dtype=np.int32)

# Number of times to verify output for a given seed.
SEED_TEST_REPETITIONS = 6


def build_test_cases(exhaustive=False):
  # Dropout rate(s) to test.
  rate = [0.1, 0.5, 0.9] if exhaustive else [0.5]

  # User specified and non-specified cases.
  seed = [SEED, None]

  # Shape of the dropout.
  # Note that shaping the dropout such that a very large portion of
  # the input weights are dropped will fail the test criteria, as expected.
  noise_shape = [[], [DIMS[0], DIMS[1], 1]]
  if exhaustive:
    noise_shape.append([DIMS[0], 1, DIMS[2]])
    noise_shape.append([1, DIMS[1], DIMS[2]])

  # Get the cartesian product (can get very large).
  prod = itertools.product(rate, seed, noise_shape)

  test_cases = []
  for n, perm in enumerate(prod):
    test = {
        'testcase_name': ' Case: %3d' % n,
        'rate': perm[0],
        'seed': perm[1],
        'noise_shape': perm[2]
    }

    test_cases.append(test)

  return test_cases


# Default is not to test every combination.
TEST_CASES = build_test_cases()


class PopnnRandomDropoutTest(test_util.TensorFlowTestCase,
                             parameterized.TestCase):
  @staticmethod
  def _ipu_dropout(w, rate, seed, noise_shape, ref):
    output = ipu.ops.rand_ops.dropout(w,
                                      rate=rate,
                                      seed=seed,
                                      noise_shape=noise_shape,
                                      ref=ref)
    return [output]

  @staticmethod
  def _setup_test(f):
    with ops.device('cpu'):
      input_data = array_ops.placeholder(np.float32, DIMS)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(f, inputs=[input_data])

      cfg = ipu.config.IPUConfig()
      cfg.auto_select_ipus = 1
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()

      return r, input_data

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testInvalidNoiseShape(self):
    in_data = np.random.rand(16, 8, 16)
    seed = np.array([12, 34], dtype=np.int32)

    with sl.Session() as sess:
      with self.assertRaisesRegex(ValueError, "must equal the rank of x."):

        def _wrong_length(w):
          return self._ipu_dropout(w, 0.5, seed, [1], False)

        r, input_data = self._setup_test(_wrong_length)
        _ = sess.run(r, {input_data: in_data})

      with self.assertRaisesRegex(ValueError, "Dimension mismatch"):

        def _wrong_dims(w):
          return self._ipu_dropout(w, 0.5, seed, [8, 1, 16], False)

        r, input_data = self._setup_test(_wrong_dims)
        _ = sess.run(r, {input_data: in_data})

  @parameterized.named_parameters(*TEST_CASES)
  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testDropout(self, rate, seed, noise_shape):
    def _run_dropout(w):
      return self._ipu_dropout(w, rate, seed, noise_shape, False)

    r, input_data = self._setup_test(_run_dropout)

    with sl.Session() as sess:
      in_data = np.random.rand(*DIMS)
      result = sess.run(r, {input_data: in_data})
      percent_kept = np.count_nonzero(
          np.array(result)) / np.count_nonzero(in_data)

      # There's a considerable amount for randomness so we have a reasonably
      # large dimensionality of test data to make sure the error is smaller.
      is_roughly_close = abs(percent_kept - (1.0 - rate))

      # The observed error is actually a lot less than this (>1%) but we don't
      # want to cause random regressions and 3% is probably still acceptable
      # for any outlier randoms.
      self.assertTrue(is_roughly_close < THRESHOLD)

  @parameterized.named_parameters(*TEST_CASES)
  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testUserSeed(self, rate, seed, noise_shape):
    # When the seed is None, we aren't testing user seeds.
    if seed is None:
      return

    def _run_dropout(w):
      return self._ipu_dropout(w, rate, seed, noise_shape, False)

    r, input_data = self._setup_test(_run_dropout)

    with sl.Session() as sess:
      in_data = np.random.rand(*DIMS)

      # For a given output, verify that each subsequent output is equal to it.
      first_result = None
      for _ in range(SEED_TEST_REPETITIONS):
        result = sess.run(r, {input_data: in_data})

        if first_result is None:
          first_result = result
          continue

        self.assertAllEqual(first_result, result)

  @parameterized.named_parameters(*TEST_CASES)
  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testDropoutBackwardPass(self, rate, seed, noise_shape):
    def _run_dropout(w):
      output = self._ipu_dropout(w, rate, seed, noise_shape, True)

      largest = output
      cost = math_ops.square(largest)

      opt = gradient_descent.GradientDescentOptimizer(learning_rate=0.1)
      gradients = opt.compute_gradients(cost, w)

      return [output, gradients]

    r, input_data = self._setup_test(_run_dropout)

    with sl.Session() as sess:
      in_data = np.random.rand(*DIMS)
      result = sess.run(r, {input_data: in_data})

      dropout_out = result[0]
      gradients = result[1][0][0]

      # Check we have the same number of zeros.
      self.assertAllEqual(np.count_nonzero(dropout_out),
                          np.count_nonzero(gradients))

  @parameterized.named_parameters(*TEST_CASES)
  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testScaling(self, rate, seed, noise_shape):
    def _run_dropout(w):
      return self._ipu_dropout(w, rate, seed, noise_shape, False)

    r, input_data = self._setup_test(_run_dropout)

    with sl.Session() as sess:
      in_data = np.ones(DIMS)
      [result] = sess.run(r, {input_data: in_data})

      kept_values = result[np.nonzero(result)]
      expected_kept_values = 1 / (1 - rate) * np.ones(kept_values.shape)

      self.assertAllClose(kept_values, expected_kept_values)


if __name__ == "__main__":
  googletest.main()
