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

import math
import numpy as np

from absl.testing import parameterized

from tensorflow.python.ipu import test_utils as tu
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python import ipu
from tensorflow.python.ipu.ops import within_replica_ops
from tensorflow.python.ipu.ops import within_replica_ops_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


def dtype_test_cases():
  return ("F32", np.float32), ("F16", np.float16), ("I32", np.int32)


def within_replica_ops_test_cases():
  reduce_scatter_with_add = lambda inputs: within_replica_ops.reduce_scatter(
      inputs, op="COLLECTIVE_OP_ADD")
  all_reduce_with_add = lambda inputs: within_replica_ops.all_reduce(
      inputs, op="COLLECTIVE_OP_ADD")
  test_cases = ("AllGather", within_replica_ops.all_gather),\
               ("ReduceScatter", reduce_scatter_with_add),\
               ("AllReduce", all_reduce_with_add)
  return test_cases


def reduce_op_numeric_test_cases():
  test_cases = \
      ("Add", "COLLECTIVE_OP_ADD", sum),\
      ("Mul", "COLLECTIVE_OP_MUL", np.prod),\
      ("Min", "COLLECTIVE_OP_MIN", min),\
      ("Max", "COLLECTIVE_OP_MAX", max)

  return test_cases


def reduce_op_logical_test_cases():
  test_cases = \
      ("And", "COLLECTIVE_OP_LOGICAL_AND", all),\
      ("Or", "COLLECTIVE_OP_LOGICAL_OR", any)

  return test_cases


def create_sharded_values(values, dtype, zero_scalar):  #pylint: disable=missing-type-doc,missing-param-doc
  """ Utility for sharding each value in values,
  so value[i] is on shard i.

  zero_scalar is a 0 valued param and is needed to prevent the
  sharding from being lost, as sharding constants
  is a bit flakey.
  """
  sharded_constants = []

  zero_scalar = math_ops.cast(zero_scalar, dtype)

  for i, value in enumerate(values):
    with ipu.scopes.ipu_shard(i):
      array_value = np.asarray(value)
      constant = constant_op.constant(array_value,
                                      shape=array_value.shape,
                                      dtype=dtype)
      sharded_constants.append(constant + zero_scalar)
  return sharded_constants


def ndarrays_to_lists(ndarrays):  #pylint: disable=missing-type-doc,missing-param-doc
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
    self._zero_val = constant_op.constant(0, shape=[], dtype=np.float32)

  @parameterized.named_parameters(*dtype_test_cases())
  @tu.test_uses_ipus(num_ipus=4)
  def testAllGather(self, dtype):
    self._cfg.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    def my_net(zero_val):
      shard0, shard1, shard2, shard3 = create_sharded_values(
          [[1, 1], [2, 2], [3, 3], [4, 4]], dtype, zero_val)
      gathered = within_replica_ops.all_gather(
          [shard0, shard1, shard2, shard3])
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[self._zero_val])

    with session.Session() as sess:
      gathered = sess.run(res)

      self.assertEqual(type(gathered), list)

      shard_count = 4
      expected_gathered_data = [1, 1, 2, 2, 3, 3, 4, 4]
      self.assertCountEqual(ndarrays_to_lists(gathered),
                            [expected_gathered_data] * shard_count)

  @tu.test_uses_ipus(num_ipus=2)
  def testAllGatherGrad(self):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    def my_net(zero_val):
      shard0, shard1 = create_sharded_values([[1, 1], [2, 2]], np.float32,
                                             zero_val)

      with backprop.GradientTape() as tape:
        tape.watch(shard0)
        tape.watch(shard1)

        gathered = within_replica_ops.all_gather([shard0, shard1])
        lossA = math_ops.reduce_mean(gathered[0]**2)
        lossB = math_ops.reduce_mean(gathered[1]**2)
        loss = lossA + lossB
      grad = tape.gradient(loss, [shard0, shard1])
      return grad

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[self._zero_val])

    with session.Session() as sess:
      grad_gathered = sess.run(res)
      self.assertEqual(type(grad_gathered), list)

      # gathered = [[1, 1, 2, 2], [1, 1, 2, 2]]
      # dLoss/dGathered = 0.5*gathered
      # dLoss/dShard = reduce_scatter([[0.5, 0.5, 1, 1], [0.5, 0.5, 1, 1], op=ADD)
      expected_grad = [[1, 1], [2, 2]]
      self.assertCountEqual(ndarrays_to_lists(grad_gathered), expected_grad)

  @tu.test_uses_ipus(num_ipus=4)
  def testAllGatherMixedSizes(self):
    self._cfg.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    def my_net(zero_val):
      shard0, shard1, shard2, shard3 = create_sharded_values(
          [[1, 1], [2, 2], [3, 3], [4, 4]], np.float32, zero_val)
      empty_shard0 = shard0[0:0]
      wide_shard2 = array_ops.concat([shard2, shard2], axis=0)
      gathered = within_replica_ops.all_gather(
          [empty_shard0, shard1, wide_shard2, shard3])
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[self._zero_val])

    with session.Session() as sess:
      gathered = sess.run(res)
      self.assertEqual(type(gathered), list)

      shard_count = 4
      expected_gathered_data = [2, 2, 3, 3, 3, 3, 4, 4]
      self.assertCountEqual(ndarrays_to_lists(gathered),
                            [expected_gathered_data] * shard_count)

  @tu.test_uses_ipus(num_ipus=2)
  def testAllGatherConcatDefault(self):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    def my_net(zero_val):
      # Values with shape (2,3)
      val1 = [[1, 1, 1], [2, 2, 2]]
      val2 = [[3, 3, 3], [4, 4, 4]]
      shard0, shard1 = create_sharded_values([val1, val2], np.float32,
                                             zero_val)
      # Test that using a tensor of rank > 1 creates a
      # tensor of equal rank where the elemts are concatenated
      # across axis 0.
      gathered = within_replica_ops.all_gather([shard0, shard1])
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[self._zero_val])

    with session.Session() as sess:
      gathered = sess.run(res)

      shard_count = 2
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

    def my_net(zero_val):
      # Values with shape (1, 2, 3) and (1, 3, 3) respectively
      val1 = [[[1, 1, 1], [2, 2, 2]]]
      val2 = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
      shard0, shard1 = create_sharded_values([val1, val2], np.float32,
                                             zero_val)
      # Test that we can use a custom axis when concatenating results
      # for tensors of rank > 1.
      gathered = within_replica_ops.all_gather([shard0, shard1], axis=1)
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[self._zero_val])

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

    def my_net(zero_val):
      shard0, shard1 = create_sharded_values([1, 2], np.int32, zero_val)
      gathered = within_replica_ops.all_gather([shard0, shard1], axis=0)
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[self._zero_val])

    with session.Session() as sess:
      gathered = sess.run(res)

      shard_count = 2
      expected_gathered_data = [1, 2]
      self.assertCountEqual(ndarrays_to_lists(gathered),
                            [expected_gathered_data] * shard_count)

  @tu.test_uses_ipus(num_ipus=2)
  def testAllGatherInvalidAxis(self):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    def my_net(zero_val):
      # Values with shape (1, 2, 3) and (1, 3, 3) respectively
      val1 = [[[1, 1, 1], [2, 2, 2]]]
      val2 = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
      shard0, shard1 = create_sharded_values([val1, val2], np.float32,
                                             zero_val)
      # Concatenating requires that all dimensions, except the one specified by
      # axis, be the same. Since that's not true here an exception should be
      # thrown.
      gathered = within_replica_ops.all_gather([shard0, shard1], axis=0)
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      self.assertRaises(ValueError,
                        ipu.ipu_compiler.compile,
                        my_net,
                        inputs=[self._zero_val])

  @tu.test_uses_ipus(num_ipus=4)
  def testAllGatherDifferentRanks(self):
    self._cfg.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    def my_net(zero_val):
      val1 = [1, 1]
      val2 = [[1], [1]]
      val3 = [[[1, 1, 1]], [[1, 1, 1]]]
      val4 = [[1], [1]]
      shard0, shard1, shard2, shard3 = create_sharded_values(
          [val1, val2, val3, val4], np.float32, zero_val)
      # Should throw an exception since each input is of a different rank.
      gathered = within_replica_ops.all_gather(
          [shard0, shard1, shard2, shard3])
      return gathered

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      self.assertRaises(ValueError,
                        ipu.ipu_compiler.compile,
                        my_net,
                        inputs=[self._zero_val])

  @parameterized.named_parameters(*within_replica_ops_test_cases())
  @tu.test_uses_ipus(num_ipus=2)
  def testOpShardOrder(self, within_replica_op):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    def my_net(zero_val):
      shard0, shard1 = create_sharded_values([[1, 1], [2, 2]], np.int32,
                                             zero_val)
      result = within_replica_op([shard1, shard0])
      return result

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[self._zero_val])

    with session.Session() as sess:
      # Should throw an error since the order of ops passed to all_gather
      # is not in shard order.
      self.assertRaises(errors.FailedPreconditionError, sess.run, res)

  @parameterized.named_parameters(*within_replica_ops_test_cases())
  def testOpMixedTypeShards(self, within_replica_op):
    def my_net(zero_val):
      shard0_int, shard1_int = create_sharded_values([[1, 1], [2, 2]],
                                                     np.int32, zero_val)
      shard0_float = math_ops.cast(shard0_int, np.float32)
      result = within_replica_op([shard0_float, shard1_int])
      return result

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      # Should throw an error since we're trying to gather tensors of
      # different types.
      self.assertRaises(ValueError,
                        ipu.ipu_compiler.compile,
                        my_net,
                        inputs=[self._zero_val])

  @parameterized.named_parameters(*within_replica_ops_test_cases())
  def testOpUniqueArgs(self, within_replica_op):
    def my_net(zero_val):
      val1 = [1, 1]
      val2 = [2, 2]
      shard0, shard1 = create_sharded_values([val1, val2], np.int32, zero_val)  #pylint: disable=unused-variable
      result = within_replica_op([shard0, shard0])
      return result

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      # Should throw an error since we're trying to reuse the same tensors.
      self.assertRaises(ValueError,
                        ipu.ipu_compiler.compile,
                        my_net,
                        inputs=[self._zero_val])


class CommonReduction:
  # Putting the reduction tests in a namespace so only their subclasses are
  # instantiated, otherwise the Tests baseclass would also be executed.
  @test_util.deprecated_graph_mode_only
  class Tests(test_util.TensorFlowTestCase, parameterized.TestCase):
    def setUp(self):
      self._cfg = ipu.config.IPUConfig()
      self._zero_val = constant_op.constant(0, shape=[], dtype=np.float32)

    @parameterized.named_parameters(*dtype_test_cases())
    @tu.test_uses_ipus(num_ipus=2)
    def testReduction(self, dtype):
      self._cfg.auto_select_ipus = 2
      tu.add_hw_ci_connection_options(self._cfg)
      self._cfg.configure_ipu_system()

      val1 = [1, 1]
      val2 = [2, 2]

      def my_net(zero_val):
        shard0, shard1 = create_sharded_values([val1, val2], dtype, zero_val)
        reduced = self._reduce_op([shard0, shard1], op="COLLECTIVE_OP_ADD")
        return reduced

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        # Same number of elements as ipus, so each ipu will recieve
        # the reduction 1+2.
        res = ipu.ipu_compiler.compile(my_net, inputs=[self._zero_val])

      with session.Session() as sess:
        reduced = sess.run(res)
        self.assertEqual(type(reduced), list)

        expected_reduction = self.reduce(sum, val1, val2)
        self.assertCountEqual(ndarrays_to_lists(reduced), expected_reduction)

    @parameterized.named_parameters(*reduce_op_numeric_test_cases())
    @tu.test_uses_ipus(num_ipus=2)
    def testNumericOps(self, ipu_reduce_op, py_reduce_op):
      self._cfg.auto_select_ipus = 2
      tu.add_hw_ci_connection_options(self._cfg)
      self._cfg.configure_ipu_system()

      val1 = [1, 1]
      val2 = [2, 2]

      def my_net(zero_val):
        shard0, shard1 = create_sharded_values([val1, val2], np.float32,
                                               zero_val)
        reduced = self._reduce_op([shard0, shard1], op=ipu_reduce_op)
        return reduced

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(my_net, inputs=[self._zero_val])

      with session.Session() as sess:
        reduced = sess.run(res)

        expected_reduction = self.reduce(py_reduce_op, val1, val2)
        self.assertCountEqual(ndarrays_to_lists(reduced), expected_reduction)

    @parameterized.named_parameters(*reduce_op_logical_test_cases())
    @tu.test_uses_ipus(num_ipus=2)
    def testLogicalOps(self, ipu_reduce_op, py_reduce_op):
      self._cfg.auto_select_ipus = 2
      tu.add_hw_ci_connection_options(self._cfg)
      self._cfg.configure_ipu_system()

      def my_net(val):
        with ipu.scopes.ipu_shard(0):
          a = math_ops.greater_equal(val, 2)
        with ipu.scopes.ipu_shard(1):
          b = math_ops.greater_equal(val, 1)
        # Reduction with args [0, 1] and [1, 1]
        reduced = self._reduce_op([a, b], op=ipu_reduce_op)
        return reduced

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        val = constant_op.constant([1, 2], shape=[2], dtype=np.int32)
        res = ipu.ipu_compiler.compile(my_net, inputs=[val])

      with session.Session() as sess:
        reduced = sess.run(res)

        expected_reduction = self.reduce(py_reduce_op, [0, 1], [1, 1])
        self.assertCountEqual(ndarrays_to_lists(reduced), expected_reduction)

    @tu.test_uses_ipus(num_ipus=4)
    def testMoreElementsThanShards(self):
      self._cfg.auto_select_ipus = 4
      tu.add_hw_ci_connection_options(self._cfg)
      self._cfg.configure_ipu_system()

      val1 = [1, 1, 1, 1, 1]
      val2 = [2, 2, 2, 2, 2]
      val3 = [3, 3, 3, 3, 3]
      val4 = [4, 4, 4, 4, 4]

      def my_net(zero_val):
        shard0, shard1, shard2, shard3 = create_sharded_values(
            [val1, val2, val3, val4], np.int32, zero_val)
        # We have more elements than ipus here so the reductions
        # won't be evenly distributed among the IPUs.
        reduced = self._reduce_op([shard0, shard1, shard2, shard3],
                                  op="COLLECTIVE_OP_ADD")
        return reduced

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(my_net, inputs=[self._zero_val])

      with session.Session() as sess:
        reduced = sess.run(res)

        expected_reduction = self.reduce(sum, val1, val2, val3, val4)
        self.assertCountEqual(ndarrays_to_lists(reduced), expected_reduction)

    @tu.test_uses_ipus(num_ipus=4)
    def testFewerElementsThanShards(self):
      self._cfg.auto_select_ipus = 4
      tu.add_hw_ci_connection_options(self._cfg)
      self._cfg.configure_ipu_system()

      val1 = [1, 1]
      val2 = [2, 2]
      val3 = [3, 3]
      val4 = [4, 4]

      def my_net(zero_val):
        shard0, shard1, shard2, shard3 = create_sharded_values(
            [val1, val2, val3, val4], np.int32, zero_val)
        # Fewer elements than ipus, so only 2 ipus will recieve the reduction
        # while the others will be padded with 0.
        reduced = self._reduce_op([shard0, shard1, shard2, shard3],
                                  op="COLLECTIVE_OP_ADD")
        return reduced

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(my_net, inputs=[self._zero_val])

      with session.Session() as sess:
        reduced = sess.run(res)

        expected_reduction = self.reduce(sum, val1, val2, val3, val4)
        self.assertCountEqual(ndarrays_to_lists(reduced), expected_reduction)

    @tu.test_uses_ipus(num_ipus=2)
    def testMixedSizes(self):
      self._cfg.auto_select_ipus = 2
      tu.add_hw_ci_connection_options(self._cfg)
      self._cfg.configure_ipu_system()

      def my_net(zero_val):
        shard0, shard1 = create_sharded_values([[2], [1, 1]], np.int32,
                                               zero_val)
        # Reductions with mixed tensor sizes should behave as if the smaller tensor
        # is padded with 0s to match the size of the largest. In this case we should
        # be doing the equivalent of [2,0] + [1,1]
        reduced = self._reduce_op([shard0, shard1], op="COLLECTIVE_OP_ADD")
        return reduced

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(my_net, inputs=[self._zero_val])

      with session.Session() as sess:
        reduced = sess.run(res)

        expected_reduction = self.reduce(sum, [2, 0], [1, 1])
        self.assertCountEqual(ndarrays_to_lists(reduced), expected_reduction)

    @tu.test_uses_ipus(num_ipus=2)
    def testScalar(self):
      self._cfg.auto_select_ipus = 2
      tu.add_hw_ci_connection_options(self._cfg)
      self._cfg.configure_ipu_system()

      def my_net(zero_val):
        shard0, shard1 = create_sharded_values([1, 2], np.int32, zero_val)
        # We expect the scalar input to be treated as a tensor of shape [1] so
        # we're doing [1] + [2]]
        reduced = self._reduce_op([shard0, shard1], op="COLLECTIVE_OP_ADD")
        return reduced

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(my_net, inputs=[self._zero_val])

      with session.Session() as sess:
        reduced = sess.run(res)

        expected_reduction = self.reduce(sum, [1], [2])
        self.assertCountEqual(reduced, expected_reduction)

    @tu.test_uses_ipus(num_ipus=2)
    def testRank1Only(self):
      self._cfg.auto_select_ipus = 2
      tu.add_hw_ci_connection_options(self._cfg)
      self._cfg.configure_ipu_system()

      rank3_val1 = [[[1, 1, 1], [2, 2, 2]]]
      rank3_val2 = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]]

      def my_net(zero_val):
        shard0, shard1 = create_sharded_values([rank3_val1, rank3_val2],
                                               np.int32, zero_val)
        reduced = self._reduce_op([shard0, shard1], op="COLLECTIVE_OP_ADD")
        return reduced

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        self.assertRaises(ValueError,
                          ipu.ipu_compiler.compile,
                          my_net,
                          inputs=[self._zero_val])


@test_util.deprecated_graph_mode_only
class ReduceScatterWithinReplicaTest(CommonReduction.Tests):
  def setUp(self):
    super().setUp()
    self._reduce_op = within_replica_ops.reduce_scatter

  def reduce(self, op, *shard_vals):  #pylint: disable=missing-type-doc,missing-param-doc
    """ Reduce shard_vals as a reduce_scatter would.
    """
    shard_count = len(shard_vals)
    reduced = [op(vals) for vals in zip(*shard_vals)]
    size = math.ceil(len(reduced) / shard_count)

    output = [[] for i in range(shard_count)]
    for i in range(size):  #pylint: disable=unused-variable
      for shard in range(shard_count):

        if reduced:
          output[shard].append(reduced.pop())
        else:
          # Pad if we dont have enough values.
          output[shard].append(0)

    return output

  @tu.test_uses_ipus(num_ipus=2)
  def testGrad(self):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    def my_net(zero_val):
      shard0, shard1 = create_sharded_values([[1, 1], [2, 2]], np.float32,
                                             zero_val)

      with backprop.GradientTape() as tape:
        tape.watch(shard0)
        tape.watch(shard1)

        reduced = within_replica_ops.reduce_scatter([shard0, shard1],
                                                    op="COLLECTIVE_OP_ADD")
        lossA = math_ops.reduce_mean(reduced[0]**2)
        lossB = math_ops.reduce_mean(reduced[1]**2)
        loss = lossA + lossB
      grad = tape.gradient(loss, [shard0, shard1])
      return grad

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[self._zero_val])

    with session.Session() as sess:
      grad = sess.run(res)
      self.assertEqual(type(grad), list)

      # reduced = [[3.0], [3.0]]
      # dLoss/dReduced = 2*reduced
      # dLoss/dShard = all_gather([[6.0], [6.0]])
      shard_count = 2
      self.assertCountEqual(ndarrays_to_lists(grad),
                            [[6.0, 6.0]] * shard_count)


@test_util.deprecated_graph_mode_only
class AllReduceWithinReplicaTest(CommonReduction.Tests):
  def setUp(self):
    super().setUp()
    self._reduce_op = within_replica_ops.all_reduce

  def reduce(self, op, *shard_vals):  #pylint: disable=missing-type-doc,missing-param-doc
    """ Reduce shard_vals as a all_reduce would.
    """
    shard_count = len(shard_vals)
    reduced = [op(vals) for vals in zip(*shard_vals)]
    return [reduced] * shard_count

  @tu.test_uses_ipus(num_ipus=2)
  def testGrad(self):
    self._cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(self._cfg)
    self._cfg.configure_ipu_system()

    def my_net(zero_val):
      shard0, shard1 = create_sharded_values([[1, 1], [2, 2]], np.float32,
                                             zero_val)

      with backprop.GradientTape() as tape:
        tape.watch(shard0)
        tape.watch(shard1)

        reduced = within_replica_ops.all_reduce([shard0, shard1],
                                                op="COLLECTIVE_OP_ADD")
        lossA = math_ops.reduce_mean(reduced[0]**2)
        lossB = math_ops.reduce_mean(reduced[1]**2)
        loss = lossA + lossB
      grad = tape.gradient(loss, [shard0, shard1])
      return grad

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[self._zero_val])

    with session.Session() as sess:
      grad = sess.run(res)
      self.assertEqual(type(grad), list)

      # reduced = [[3.0, 3.0], [3.0, 3.0]]
      # dLoss/dReduced = reduced
      # dLoss/dShard = all_reduce([[3.0, 3.0], [3.0, 3.0]], op=original_op)
      shard_count = 2
      self.assertCountEqual(ndarrays_to_lists(grad),
                            [[6.0, 6.0]] * shard_count)


if __name__ == "__main__":
  test.main()
