# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.platform import googletest


class TestCandidateSampler(xla_test.XLATestCase):  # pylint: disable=abstract-method
  UNIQUE_TEST_CASES = [[1, 100], [10, 100], [90, 100], [100, 100]]
  TEST_CASES = UNIQUE_TEST_CASES + [[1000, 100]]

  def check_compiles(self, k, N, unique=False, dist="uniform", seed=None):
    with self.session() as sess:
      np.random.seed(42)
      true_classes = np.arange(N).reshape([1, -1])

      if dist == "uniform":
        sampling_op = candidate_sampling_ops.uniform_candidate_sampler
      elif dist == "log_uniform":
        sampling_op = candidate_sampling_ops.log_uniform_candidate_sampler
      else:
        raise Exception(f"generate_op: dist {dist} not supported.")

      def body(x):
        return sampling_op(true_classes=x,
                           num_true=true_classes.shape[1],
                           num_sampled=k,
                           unique=unique,
                           range_max=N,
                           seed=seed)

      with ops.device('cpu'):
        inp = array_ops.placeholder(np.int64, true_classes.shape)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        ipu_op = ipu.ipu_compiler.compile(body, inputs=[inp])

      sess.run(ipu_op, feed_dict={inp: true_classes})

  @test_util.deprecated_graph_mode_only
  def testUniformSamplerWithReplacement(self):
    # Expectation for uniform is always k/p
    self.check_compiles(25, 100, dist="uniform", unique=False)

  @test_util.deprecated_graph_mode_only
  def testUniformSamplerWithoutReplacement(self):
    self.check_compiles(25, 100, dist="uniform", unique=True)

  @test_util.deprecated_graph_mode_only
  def testLogUniformSamplerWithReplacement(self):
    self.check_compiles(25, 100, dist="log_uniform", unique=False)

  @test_util.deprecated_graph_mode_only
  def testLogUniformSamplerWithoutReplacement(self):
    self.check_compiles(25, 100, dist="log_uniform", unique=True)

  @test_util.deprecated_graph_mode_only
  def testFixedSeed(self):
    self.check_compiles(25, 100, dist="log_uniform", unique=False, seed=42)


if __name__ == "__main__":
  googletest.main()
