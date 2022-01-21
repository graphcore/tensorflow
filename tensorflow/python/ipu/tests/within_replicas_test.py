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

import operator
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


def dtype_test_cases():
  return ("F32", np.float32), ("F16", np.float16), ("I32", np.int32)


def within_replica_ops_test_cases():
  reduce_scatter_with_add = lambda inputs: within_replica_ops.reduce_scatter(
      inputs, op="COLLECTIVE_OP_ADD")
  test_cases = ("AllGather", within_replica_ops.all_gather),\
               ("ReduceScatter", reduce_scatter_with_add)
  return test_cases


def reduce_op_numeric_test_cases():
  test_cases = \
      ("Add", "COLLECTIVE_OP_ADD", operator.add),\
      ("Mul", "COLLECTIVE_OP_MUL", operator.mul),\
      ("Min", "COLLECTIVE_OP_MIN", min),\
      ("Max", "COLLECTIVE_OP_MAX", max)

  return test_cases


def reduce_op_logical_test_cases():
  test_cases = \
      ("And", "COLLECTIVE_OP_LOGICAL_AND", lambda x, y: x and y),\
      ("Or", "COLLECTIVE_OP_LOGICAL_OR", lambda x, y: x or y)

  return test_cases


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

  @parameterized.named_parameters(*dtype_test_cases())
  @tu.test_uses_ipus(num_ipus=4)
  def testAllGather(self, dtype):
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
  def testAllGatherMixedSizes(self):
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
  def testAllGatherConcatDefault(self):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    shard_count = 2

    def my_net(val):
      shard0, shard1 = increment_val_over_shards(shard_count, val)
      # Test that using a tensor of rank > 1 creates a
      # tensor of equal rank where the elemts are concatenated
      # across axis 0.
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
  def testAllGatherConcatAxis(self):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    def my_net(val1, val2):
      with ipu.scopes.ipu_shard(0):
        a = val1 + 1
      with ipu.scopes.ipu_shard(1):
        b = val2 + 7
      # Test that we can use a custom axis when concatenating results
      # for tensors of rank > 1.
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
  def testAllGatherScalar(self):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    shard_count = 2

    def my_net(zero_val):
      shard0, shard1 = increment_val_over_shards(shard_count, zero_val)
      gathered = within_replica_ops.all_gather([shard0, shard1], axis=0)
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      zero_val = constant_op.constant(0, shape=[], dtype=np.int32)
      res = ipu.ipu_compiler.compile(my_net, inputs=[zero_val])

    with session.Session() as sess:
      gathered = sess.run(res)

      expected_gathered_data = [1, 2]
      self.assertCountEqual(ndarrays_to_lists(gathered),
                            [expected_gathered_data] * shard_count)

  @tu.test_uses_ipus(num_ipus=2)
  def testAllGatherInvalidAxis(self):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    def my_net(val1, val2):
      with ipu.scopes.ipu_shard(0):
        a = val1 + 1
      with ipu.scopes.ipu_shard(1):
        b = val2 + 1
      # Concatenating requires that all dimensions, except the one specified by
      # axis, be the same. Since that's not true here an exception should be
      # thrown.
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
  def testAllGatherDifferentRanks(self):
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

  @parameterized.named_parameters(*dtype_test_cases())
  @tu.test_uses_ipus(num_ipus=2)
  def testReduceScatter(self, dtype):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    shard_count = 2

    def my_net(zero_val):
      shard0, shard1 = increment_val_over_shards(shard_count, zero_val)
      reduced = within_replica_ops.reduce_scatter([shard0, shard1],
                                                  op="COLLECTIVE_OP_ADD")
      return reduced

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      # Same number of elements as ipus, so each ipu will recieve
      # the reduction 1+2.
      zero_val = constant_op.constant(0, shape=[2], dtype=dtype)
      res = ipu.ipu_compiler.compile(my_net, inputs=[zero_val])

    with session.Session() as sess:
      reduced = sess.run(res)
      self.assertEqual(type(reduced), list)

      # 2 elements since we have 2 shards.
      self.assertCountEqual(reduced, [[3], [3]])

  @parameterized.named_parameters(*reduce_op_numeric_test_cases())
  @tu.test_uses_ipus(num_ipus=2)
  def testReduceScatterNumericOps(self, ipu_reduce_op, py_reduce_op):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    shard_count = 2

    def my_net(zero_val):
      shard0, shard1 = increment_val_over_shards(shard_count, zero_val)
      # Reduce scatter [1, 1], [2, 2] using the given op.
      reduced = within_replica_ops.reduce_scatter([shard0, shard1],
                                                  op=ipu_reduce_op)
      return reduced

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      zero_val = constant_op.constant(0, shape=[2], dtype=np.float32)
      res = ipu.ipu_compiler.compile(my_net, inputs=[zero_val])

    with session.Session() as sess:
      reduced = sess.run(res)

      expected_result = [
          py_reduce_op(*ab) for ab in zip([1.0, 1.0], [2.0, 2.0])
      ]
      self.assertCountEqual(reduced, expected_result)

  @parameterized.named_parameters(*reduce_op_logical_test_cases())
  @tu.test_uses_ipus(num_ipus=2)
  def testReduceScatterLogicalOps(self, ipu_reduce_op, py_reduce_op):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    def my_net(val):
      with ipu.scopes.ipu_shard(0):
        a = math_ops.greater_equal(val, 2)
      with ipu.scopes.ipu_shard(1):
        b = math_ops.greater_equal(val, 1)
      # Reduction with args [0, 1] and [1, 1]
      reduced = within_replica_ops.reduce_scatter([a, b], op=ipu_reduce_op)
      return reduced

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      val = constant_op.constant([1, 2], shape=[2], dtype=np.int32)
      res = ipu.ipu_compiler.compile(my_net, inputs=[val])

    with session.Session() as sess:
      reduced = sess.run(res)

      expected_result = [py_reduce_op(*ab) for ab in zip([0, 1], [1, 1])]
      self.assertCountEqual(reduced, expected_result)

  @tu.test_uses_ipus(num_ipus=4)
  def testReduceScatterMoreElementsThanShards(self):
    self._cfg.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    shard_count = 4

    def my_net(zero_val):
      shard0, shard1, shard2, shard3 = increment_val_over_shards(
          shard_count, zero_val)
      reduced = within_replica_ops.reduce_scatter(
          [shard0, shard1, shard2, shard3], op="COLLECTIVE_OP_ADD")
      return reduced

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      # We have more elements than ipus here so the reductions
      # won't be evenly distributed among the IPUs.
      zero_val = constant_op.constant(0, shape=[5], dtype=np.int32)
      res = ipu.ipu_compiler.compile(my_net, inputs=[zero_val])

    with session.Session() as sess:
      reduced = sess.run(res)

      # 4 elements since we have 4 shards.
      self.assertCountEqual(ndarrays_to_lists(reduced),
                            [[10, 0], [10, 10], [10, 0], [10, 0]])

  @tu.test_uses_ipus(num_ipus=4)
  def testReduceScatterFewerElementsThanShards(self):
    self._cfg.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    shard_count = 4

    def my_net(zero_val):
      shard0, shard1, shard2, shard3 = increment_val_over_shards(
          shard_count, zero_val)
      reduced = within_replica_ops.reduce_scatter(
          [shard0, shard1, shard2, shard3], op="COLLECTIVE_OP_ADD")
      return reduced

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      # Fewer elements than ipus, so only 2 ipus will recieve the reduction
      # while the others will be padded with 0.
      zero_val = constant_op.constant(0, shape=[2], dtype=np.int32)
      res = ipu.ipu_compiler.compile(my_net, inputs=[zero_val])

    with session.Session() as sess:
      reduced = sess.run(res)
      self.assertCountEqual(reduced, [[0], [10], [10], [0]])

  @tu.test_uses_ipus(num_ipus=2)
  def testReduceScatterMixedSizes(self):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    def my_net(val):
      with ipu.scopes.ipu_shard(0):
        a = val + 1
      with ipu.scopes.ipu_shard(1):
        b = array_ops.concat([val, val], axis=0)
      # Reductions with mixed tensor sizes should behave as if the smaller tensor
      # is padded with 0s to match the size of the largest. In this case we should
      # be doing the equivalent of [2,0] + [1,1]
      reduced = within_replica_ops.reduce_scatter([a, b],
                                                  op="COLLECTIVE_OP_ADD")
      return reduced

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      val = constant_op.constant(1, shape=[1], dtype=np.int32)
      res = ipu.ipu_compiler.compile(my_net, inputs=[val])

    with session.Session() as sess:
      reduced = sess.run(res)
      self.assertCountEqual(reduced, [[3], [1]])

  @tu.test_uses_ipus(num_ipus=2)
  def testReduceScatterScalar(self):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    shard_count = 2

    def my_net(zero_scalar):
      shard0, shard1 = increment_val_over_shards(shard_count, zero_scalar)
      # We expect the scalar input to be treated as a tensor of shape [1] so
      # we're doing [1] + [2]]
      reduced = within_replica_ops.reduce_scatter([shard0, shard1],
                                                  op="COLLECTIVE_OP_ADD")
      return reduced

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      zero_scalar = constant_op.constant(0, shape=[], dtype=np.int32)
      res = ipu.ipu_compiler.compile(my_net, inputs=[zero_scalar])

    with session.Session() as sess:
      reduced = sess.run(res)
      self.assertCountEqual(reduced, [[0], [3]])

  @tu.test_uses_ipus(num_ipus=2)
  def testReduceScatterRank1Only(self):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    shard_count = 2

    def my_net(zero_val):
      shard0, shard1 = increment_val_over_shards(shard_count, zero_val)
      reduced = within_replica_ops.reduce_scatter([shard0, shard1],
                                                  op="COLLECTIVE_OP_ADD")
      return reduced

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      zero_val = constant_op.constant(0, shape=[2, 2], dtype=np.int32)
      self.assertRaises(ValueError,
                        ipu.ipu_compiler.compile,
                        my_net,
                        inputs=[zero_val])

  @parameterized.named_parameters(*within_replica_ops_test_cases())
  @tu.test_uses_ipus(num_ipus=2)
  def testOpShardOrder(self, within_replica_op):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    shard_count = 2

    def my_net(val):
      shard0, shard1 = increment_val_over_shards(shard_count, val)
      result = within_replica_op([shard1, shard0])
      return result

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      val = constant_op.constant(1, shape=[2], dtype=np.int32)
      res = ipu.ipu_compiler.compile(my_net, inputs=[val])

    with session.Session() as sess:
      # Should throw an error since the order of ops passed to all_gather
      # is not in shard order.
      self.assertRaises(errors.FailedPreconditionError, sess.run, res)

  @parameterized.named_parameters(*within_replica_ops_test_cases())
  def testOpMixedTypeShards(self, within_replica_op):
    shard_count = 2

    def my_net(int_val):
      shard0_int, shard1_int = increment_val_over_shards(shard_count, int_val)
      shard0_float = math_ops.cast(shard0_int, np.float32)
      result = within_replica_op([shard0_float, shard1_int])
      return result

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      # Should throw an error since we're trying to gather tensors of
      # different types.
      int_val = constant_op.constant(1, shape=[2], dtype=np.int32)
      self.assertRaises(ValueError,
                        ipu.ipu_compiler.compile,
                        my_net,
                        inputs=[int_val])

  @parameterized.named_parameters(*within_replica_ops_test_cases())
  def testOpUniqueArgs(self, within_replica_op):
    shard_count = 2

    def my_net(val):
      shard0, shard1 = increment_val_over_shards(shard_count, val)  #pylint: disable=unused-variable
      result = within_replica_op([shard0, shard0])
      return result

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      # Should throw an error since we're trying to reuse the same tensors.
      val = constant_op.constant(1, shape=[2], dtype=np.int32)
      self.assertRaises(ValueError,
                        ipu.ipu_compiler.compile,
                        my_net,
                        inputs=[val])


if __name__ == "__main__":
  test.main()
