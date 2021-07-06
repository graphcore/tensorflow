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

from absl.testing import parameterized

from tensorflow.python.ipu.config import IPUConfig

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import googletest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


def create_n_replica_ipu_config(ipu_count):
  cfg = IPUConfig()
  cfg._profiling.profiling = False  # pylint: disable=protected-access
  cfg.optimizations.maximum_cross_replica_sum_buffer_size = 10000
  cfg.auto_select_ipus = ipu_count
  tu.add_hw_ci_connection_options(cfg)

  return cfg


def create_constant_repeating_dataset(value, shape):
  dataset = dataset_ops.Dataset.from_tensors(
      constant_op.constant(value, shape=shape))
  return dataset.repeat()


def run_body_repeatedly(body, inputs, infeed, iterations, config):
  def my_net():
    r = ipu.loops.repeat(iterations, body, inputs, infeed)
    return r

  with ipu.scopes.ipu_scope("/device:IPU:0"):
    res = ipu.ipu_compiler.compile(my_net, inputs=[])

  config.configure_ipu_system()

  with sl.Session() as sess:
    sess.run(infeed.initializer)
    result = sess.run(res)
    return result


@test_util.deprecated_graph_mode_only
class TestAssumeEqual(test_util.TensorFlowTestCase, parameterized.TestCase):
  # assume_equal_across_replicas supports copying or inplace operation depending
  # on the value of the inplace argument
  inplace_or_copy = [True, False]

  # the issue in this test can't be reproduces with 2 ipus.
  @tu.test_uses_ipus(num_ipus=4)
  def testNoDivergenceWithSlicedTensor(self):
    input_shape = [10]
    dataset = create_constant_repeating_dataset(1.0, input_shape)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def no_divergence_with_sliced_tensor(index, total,
                                         replica_dependent_value):
      def _enq_fn(to_enq):
        return outfeed_queue.enqueue(to_enq)

      # This test cant be inplace since we're using tensor slices.
      inplace = False
      safe_value = ipu.cross_replica_ops.assume_equal_across_replicas(
          replica_dependent_value[index], inplace)
      intermediate = ipu.ops.cross_replica_ops.cross_replica_sum(safe_value)
      maybe_enqueue_op = control_flow_ops.cond(
          math_ops.greater(intermediate, 0), lambda: _enq_fn(intermediate),
          lambda: control_flow_ops.no_op())  # pylint: disable=W0108

      with ops.control_dependencies([maybe_enqueue_op]):
        itermediate = array_ops.identity(intermediate)

      return index + 1, total + itermediate, replica_dependent_value

    def body(total, replica_dependent_value):
      start = constant_op.constant(0)
      _, total, _ = control_flow_ops.while_loop(
          cond=lambda i, *_: math_ops.less(i, 4),
          body=no_divergence_with_sliced_tensor,
          loop_vars=[start, total, replica_dependent_value])
      return total

    total = constant_op.constant(0, shape=input_shape, dtype=np.float32)
    iterations = 1  # we want to provide data from an infeed, not run 'body' repeatedly.
    result = run_body_repeatedly(body, [total], infeed_queue, iterations,
                                 create_n_replica_ipu_config(4))

    # 1(infeed value)*4(replicas)*4(repeats) = 16
    self.assertAllClose(result[0], np.broadcast_to(16.0, input_shape))

  @parameterized.parameters(inplace_or_copy)
  @tu.test_uses_ipus(num_ipus=2)
  def testNoDivergenceWithSingleTensor(self, inplace):

    input_shape = [2, 4]
    dataset = create_constant_repeating_dataset(1.0, input_shape)
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def no_divergence_with_single_tensor(total, replica_dependent_value):
      y = constant_op.constant(2.0, shape=input_shape, dtype=np.float32)
      divergent_true_condition = math_ops.reduce_all(
          math_ops.not_equal(replica_dependent_value, y))
      safe_true_condition = \
        ipu.ops.cross_replica_ops.assume_equal_across_replicas(
            divergent_true_condition, inplace)

      total = control_flow_ops.cond(
          safe_true_condition, lambda: ipu.ops.cross_replica_ops.
          cross_replica_sum(total + replica_dependent_value), lambda:
          constant_op.constant(0.0, shape=input_shape, dtype=np.float32))
      return total

    iterations = 2
    total = constant_op.constant(0.0, shape=input_shape, dtype=np.float32)
    result = run_body_repeatedly(no_divergence_with_single_tensor, [total],
                                 infeed_queue, iterations,
                                 create_n_replica_ipu_config(2))

    # 1(infeed value)*2(replicas) = 2
    # 2+1(infeed value)*2(replicas) = 6
    self.assertAllClose(result[0], np.broadcast_to(6.0, input_shape))

  @parameterized.parameters(inplace_or_copy)
  @tu.test_uses_ipus(num_ipus=2)
  def testNoDivergenceWithNestedTensors(self, inplace):

    input_shape = [2, 4]
    dataset = create_constant_repeating_dataset(1.0, input_shape)
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def no_divergence_with_multiple_tensors(total, x):
      y1 = constant_op.constant(2.0, shape=input_shape, dtype=np.float32)
      y2 = constant_op.constant(3.0, shape=input_shape, dtype=np.float32)
      y3 = constant_op.constant(4.0, shape=input_shape, dtype=np.float32)

      divergent_condition1 = math_ops.not_equal(x, y1)
      divergent_condition2 = math_ops.not_equal(x, y2)
      divergent_condition3 = math_ops.not_equal(x, y3)
      divergent_conditions = [
          divergent_condition1, divergent_condition2, divergent_condition3
      ]

      safe_conditions = ipu.ops.cross_replica_ops.assume_equal_across_replicas(
          divergent_conditions, inplace)
      safe_condition1, safe_condition2, safe_condition3 = safe_conditions

      true_condition = math_ops.reduce_all(
          math_ops.logical_and(
              math_ops.logical_and(safe_condition1, safe_condition2),
              safe_condition3))

      total = control_flow_ops.cond(
          true_condition,
          lambda: ipu.ops.cross_replica_ops.cross_replica_sum(total + x),
          lambda: constant_op.constant(
              0.0, shape=input_shape, dtype=np.float32))
      return total

    iterations = 2
    total = constant_op.constant(0.0, shape=input_shape, dtype=np.float32)
    result = run_body_repeatedly(no_divergence_with_multiple_tensors, [total],
                                 infeed_queue, iterations,
                                 create_n_replica_ipu_config(2))

    # 1(infeed value)*2(replicas) = 2
    # 2+1(infeed value)*2(replicas) = 6
    self.assertAllClose(result[0], np.broadcast_to(6.0, input_shape))


if __name__ == "__main__":
  googletest.main()
