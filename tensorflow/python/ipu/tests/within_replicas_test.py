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
# =============================================================================

import numpy as np

from absl.testing import parameterized

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python import ipu
from tensorflow.python.ipu.ops import within_replica_ops
from tensorflow.python.platform import test


def increment_val_over_shards(num_shards, val):
  """ Sharded increment of val, so in shard i val is incremented by i + 1.
  """
  values = []
  for i in range(num_shards):
    with ipu.scopes.ipu_shard(i):
      values.append(val + i + 1)
  return values


def ndarrays_to_lists(ndarrays):
  """ Utility for using converting ndarrays to list so the values
  can be passed into assertCountEqual, which would otherwise error with...

  ValueError: The truth value of an array with more than one element is
  ambiguous. Use a.any() or a.all()
  """
  return [ndarray.tolist() for ndarray in ndarrays]


@test_util.deprecated_graph_mode_only
class WithinReplicasTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  def setUp(self):
    self._cfg = ipu.config.IPUConfig()

  @parameterized.named_parameters(
      ("F32", np.float32),
      ("F16", np.float16),
      ("I32", np.int32),
  )
  @tu.test_uses_ipus(num_ipus=4)
  def testAllGatherWithinReplica(self, dtype):
    self._cfg.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    shard_count = 4

    def my_net(zero_val):
      shard0, shard1, shard2, shard3 = increment_val_over_shards(
          shard_count, zero_val)
      gathered = within_replica_ops.all_gather(
          [shard0, shard1, shard2, shard3])
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      zero_val = constant_op.constant(0, shape=[2], dtype=dtype)
      res = ipu.ipu_compiler.compile(my_net, inputs=[zero_val])

    with session.Session() as sess:
      gathered = sess.run(res)

      self.assertEqual(type(gathered), list)

      expected_gathered_data = [1, 1, 2, 2, 3, 3, 4, 4]
      self.assertCountEqual(ndarrays_to_lists(gathered),
                            [expected_gathered_data] * shard_count)

  @tu.test_uses_ipus(num_ipus=4)
  def testAllGatherWithinReplicaMixedSizes(self):
    self._cfg.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    shard_count = 4

    def my_net(zero_val):
      shard0, shard1, shard2, shard3 = increment_val_over_shards(
          shard_count, zero_val)
      empty_shard0 = shard0[0:0]
      wide_shard2 = array_ops.concat([shard2, shard2], axis=0)
      gathered = within_replica_ops.all_gather(
          [empty_shard0, shard1, wide_shard2, shard3])
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      zero_val = constant_op.constant(0, shape=[2], dtype=np.float32)
      res = ipu.ipu_compiler.compile(my_net, inputs=[zero_val])

    with session.Session() as sess:
      gathered = sess.run(res)
      self.assertEqual(type(gathered), list)

      expected_gathered_data = [2, 2, 3, 3, 3, 3, 4, 4]
      self.assertCountEqual(ndarrays_to_lists(gathered),
                            [expected_gathered_data] * shard_count)

  @tu.test_uses_ipus(num_ipus=2)
  def testAllGatherWithinReplicaConcatDefault(self):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    shard_count = 2

    def my_net(val):
      shard0, shard1 = increment_val_over_shards(shard_count, val)
      gathered = within_replica_ops.all_gather([shard0, shard1])
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      val = math_ops.range(6, dtype=np.int32)
      val = array_ops.reshape(val, [2, 3])
      res = ipu.ipu_compiler.compile(my_net, inputs=[val])

    with session.Session() as sess:
      gathered = sess.run(res)
      self.assertEqual(len(gathered), shard_count)
      # The default axis for concatenation should be 0, hence
      # the resulting shape should be [4,3]
      self.assertEqual(gathered[0].shape, (4, 3))
      self.assertEqual(gathered[1].shape, (4, 3))

  @tu.test_uses_ipus(num_ipus=2)
  def testAllGatherWithinReplicaConcatAxis(self):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    def my_net(val1, val2):
      with ipu.scopes.ipu_shard(0):
        a = val1 + 1
      with ipu.scopes.ipu_shard(1):
        b = val2 + 7
      gathered = within_replica_ops.all_gather([a, b], axis=1)
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      val1 = math_ops.range(6, dtype=np.int32)
      val1 = array_ops.reshape(val1, [1, 2, 3])
      val2 = math_ops.range(9, dtype=np.int32)
      val2 = array_ops.reshape(val2, [1, 3, 3])
      res = ipu.ipu_compiler.compile(my_net, inputs=[val1, val2])

    with session.Session() as sess:
      gathered = sess.run(res)
      self.assertEqual(len(gathered), 2)
      # Shape is [1,5,3] since we've concatenated using axis 1
      self.assertEqual(gathered[0].shape, (1, 5, 3))
      self.assertEqual(gathered[1].shape, (1, 5, 3))

  @tu.test_uses_ipus(num_ipus=2)
  def testAllGatherWithinReplicaScalar(self):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    shard_count = 2

    def my_net(ones_val):
      shard0, shard1 = increment_val_over_shards(shard_count, ones_val)
      gathered = within_replica_ops.all_gather([shard0, shard1], axis=0)
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      ones_val = constant_op.constant(1, shape=[], dtype=np.int32)
      res = ipu.ipu_compiler.compile(my_net, inputs=[ones_val])

    with session.Session() as sess:
      gathered = sess.run(res)
      self.assertEqual(len(gathered), shard_count)
      self.assertAllEqual(gathered[0], [2, 3])
      self.assertAllEqual(gathered[1], [2, 3])

  @tu.test_uses_ipus(num_ipus=2)
  def testAllGatherWithinReplicaInvalidAxis(self):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    def my_net(val1, val2):
      with ipu.scopes.ipu_shard(0):
        a = val1 + 1
      with ipu.scopes.ipu_shard(1):
        b = val2 + 1
      # This requires that all dimensions, except the one specified by axis,
      # be the same. Since that's not true here an exception should be thrown.
      gathered = within_replica_ops.all_gather([a, b], axis=0)
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      val1 = constant_op.constant(1, shape=[1, 3, 3], dtype=np.int32)
      val2 = constant_op.constant(2, shape=[1, 2, 3], dtype=np.int32)
      self.assertRaises(ValueError,
                        ipu.ipu_compiler.compile,
                        my_net,
                        inputs=[val1, val2])

  @tu.test_uses_ipus(num_ipus=4)
  def testAllGatherWithinReplicaDifferentRanks(self):
    self._cfg.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    def my_net(val1, val2, val3, val4):
      with ipu.scopes.ipu_shard(0):
        a = val1 + 1
      with ipu.scopes.ipu_shard(1):
        b = val2 + 2
      with ipu.scopes.ipu_shard(2):
        c = val3 + 3
      with ipu.scopes.ipu_shard(3):
        d = val4 + 4
      # Should throw an exception since each input is of a different rank.
      gathered = within_replica_ops.all_gather([a, b, c, d])
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      val1 = constant_op.constant(0, shape=[2], dtype=np.float32)
      val2 = constant_op.constant(0, shape=[2, 1], dtype=np.float32)
      val3 = constant_op.constant(0, shape=[2, 1, 3], dtype=np.float32)
      val4 = constant_op.constant(0, shape=[2, 1], dtype=np.float32)
      self.assertRaises(ValueError,
                        ipu.ipu_compiler.compile,
                        my_net,
                        inputs=[val1, val2, val3, val4])

  @tu.test_uses_ipus(num_ipus=2)
  def testAllGatherWithinReplicaShardOrder(self):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    def my_net():
      with ipu.scopes.ipu_shard(0):
        a = constant_op.constant(1, shape=[2], dtype=np.int32)
      with ipu.scopes.ipu_shard(1):
        b = constant_op.constant(2, shape=[2], dtype=np.int32)
      gathered = within_replica_ops.all_gather([b, a])
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    with session.Session() as sess:
      # Should throw an error since the order of ops passed to all_gather
      # is not in shard order.
      self.assertRaises(errors.FailedPreconditionError, sess.run, res)

  def testAllGatherWithinReplicaMixedTypeShards(self):
    def my_net():
      with ipu.scopes.ipu_shard(0):
        a = constant_op.constant(1, shape=[2], dtype=np.int32)
      with ipu.scopes.ipu_shard(1):
        b = constant_op.constant(2.0, shape=[2], dtype=np.float32)
      gathered = within_replica_ops.all_gather([a, b])
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      # Should throw an error since we're trying to gather tensors of
      # different types.
      self.assertRaises(ValueError, ipu.ipu_compiler.compile, my_net)

  def testAllGatherWithinReplicaUniqueArgs(self):
    def my_net():
      with ipu.scopes.ipu_shard(0):
        a = constant_op.constant(1, shape=[2], dtype=np.int32)
      with ipu.scopes.ipu_shard(1):
        b = constant_op.constant(2, shape=[2], dtype=np.int32)  #pylint: disable=unused-variable
      gathered = within_replica_ops.all_gather([a, a])
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      # Should throw an error since we're trying to gather tensors of
      # different types.
      self.assertRaises(ValueError, ipu.ipu_compiler.compile, my_net)


if __name__ == "__main__":
  test.main()
