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
import numpy as np

from tensorflow.python.ipu import test_utils as tu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.platform import googletest
from tensorflow.python import ipu


class TestRandomNumberGeneration(test_util.TensorFlowTestCase):
  @classmethod
  def setUpClass(cls):
    cls.session = sl.Session()
    cls.seeds = [0x01234567, 0x89abcdef]

    # The test graph
    with ipu.scopes.ipu_scope('/device:IPU:0'):
      cls.starting_seed = array_ops.placeholder(np.int32, [2])

      cls.truncated_normal = stateless_random_ops.stateless_truncated_normal(
          shape=[1000], seed=cls.starting_seed, dtype=np.float32)

      cls.normal = stateless_random_ops.stateless_random_normal(
          shape=[1000], seed=cls.starting_seed, dtype=np.float32)

      cls.uniform_float = stateless_random_ops.stateless_random_uniform(
          shape=[1000], seed=cls.starting_seed, dtype=np.float32)

      cls.uniform_int = stateless_random_ops.stateless_random_uniform(
          shape=[1000],
          seed=cls.starting_seed,
          minval=0,
          maxval=1000,
          dtype=np.int32)

    # Configure the hardware
    config = ipu.config.IPUConfig()
    tu.add_hw_ci_connection_options(config)
    config.configure_ipu_system()

  def _compare_ops(self, ops):
    refs = self.session.run(ops, {self.starting_seed: self.seeds})

    for _ in range(100):
      vals = self.session.run(ops, {self.starting_seed: self.seeds})
      for pair in zip(refs, vals):
        assert np.array_equal(pair[0], pair[1])

    reversed_seeds = list(reversed(self.seeds))
    for _ in range(100):
      vals = self.session.run(ops, {self.starting_seed: reversed_seeds})
      for pair in zip(refs, vals):
        assert not np.array_equal(pair[0], pair[1])

  # Operators alone
  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_tuncated_normal(self):
    self._compare_ops([self.truncated_normal])

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_normal(self):
    self._compare_ops([self.normal])

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_uniform_float(self):
    self._compare_ops([self.uniform_float])

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_uniform_integer(self):
    self._compare_ops([self.uniform_int])

  # Operator pairs
  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_tuncated_normal_and_normal(self):
    self._compare_ops([self.truncated_normal, self.normal])

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_tuncated_normal_and_uniform_float(self):
    self._compare_ops([self.truncated_normal, self.uniform_float])

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_tuncated_normal_and_uniform_int(self):
    self._compare_ops([self.truncated_normal, self.uniform_int])

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_normal_and_uniform_float(self):
    self._compare_ops([self.normal, self.uniform_float])

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_normal_and_uniform_int(self):
    self._compare_ops([self.normal, self.uniform_int])

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_uniform_float_and_uniform_int(self):
    self._compare_ops([self.uniform_float, self.uniform_int])

  # Operator triplets
  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_tuncated_normal_and_normal_and_uniform_float(self):
    self._compare_ops([self.truncated_normal, self.normal, self.uniform_float])

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_tuncated_normal_and_normal_and_uniform_int(self):
    self._compare_ops([self.truncated_normal, self.normal, self.uniform_int])

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_tuncated_normal_and_uniform_float_and_uniform_int(self):
    self._compare_ops(
        [self.truncated_normal, self.uniform_float, self.uniform_int])

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_normal_and_uniform_float_and_uniform_int(self):
    self._compare_ops([self.normal, self.uniform_float, self.uniform_int])

  # All operators
  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def test_all(self):
    self._compare_ops([
        self.truncated_normal, self.normal, self.uniform_float,
        self.uniform_int
    ])


if __name__ == '__main__':
  googletest.main()
