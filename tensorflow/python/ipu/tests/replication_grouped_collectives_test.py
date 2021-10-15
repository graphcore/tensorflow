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
import unittest
import numpy as np

from tensorflow.python.ipu.config import IPUConfig
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors_impl


def run_collective_ops(inputs, generate_collective_ops, num_replicas):
  with tu.ipu_session() as sess:
    dataset = dataset_ops.Dataset.from_tensor_slices(inputs)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def my_net(x):
      y = generate_collective_ops(x)
      return outfeed_queue.enqueue(y)

    def my_loop():
      return ipu.loops.repeat(1, my_net, infeed_queue=infeed_queue)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      compiled_net = ipu.ipu_compiler.compile(my_loop)

    dequeued = outfeed_queue.dequeue()

    cfg = ipu.config.IPUConfig()
    cfg.optimizations.maximum_reduce_scatter_buffer_size = 10000
    cfg.auto_select_ipus = num_replicas
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    sess.run(infeed_queue.initializer)

    try:
      sess.run(compiled_net)
      outputs = sess.run(dequeued)
      sess.run(outfeed_queue.deleter)
      return outputs
    finally:
      sess.run(infeed_queue.deleter)


@unittest.skip("T42121 - currently not supported.")
class TestReplicationGroupedCollectives(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def test_grouped_reduce_scatter(self):
    def my_collective_ops(x):
      return [
          ipu.ops.reduce_scatter_op.reduce_scatter(
              x, replication_factor=replica_group_size)
          for replica_group_size in [1, 2, 4]
      ]

    inputs = np.array(
        [
            [0., 0., 1., 2.],  # replica 0
            [0., 1., 2., 3.],  # replica 1
            [1., 2., 3., 4.],  # replica 2
            [2., 3., 4., 5.],  # replica 3
        ],
        dtype=np.float32)

    outputs = run_collective_ops(inputs, my_collective_ops, num_replicas=4)
    [group_size_1], [group_size_2], [group_size_4] = outputs

    # Group size 1.
    self.assertAllEqual(group_size_1[0], inputs[0])
    self.assertAllEqual(group_size_1[1], inputs[1])
    self.assertAllEqual(group_size_1[2], inputs[2])
    self.assertAllEqual(group_size_1[3], inputs[3])

    # Group size 2.
    self.assertAllEqual(group_size_2[0], inputs[0:2, 0:2].sum(axis=1))
    self.assertAllEqual(group_size_2[1], inputs[0:2, 2:4].sum(axis=1))
    self.assertAllEqual(group_size_2[2], inputs[2:4, 0:2].sum(axis=1))
    self.assertAllEqual(group_size_2[3], inputs[2:4, 2:4].sum(axis=1))

    # Group size 4.
    self.assertAllEqual(group_size_4[0], inputs[0:4, 0:1].sum(axis=0))
    self.assertAllEqual(group_size_4[1], inputs[0:4, 1:2].sum(axis=0))
    self.assertAllEqual(group_size_4[2], inputs[0:4, 2:3].sum(axis=0))
    self.assertAllEqual(group_size_4[3], inputs[0:4, 3:4].sum(axis=0))

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def test_grouped_all_gather(self):
    def my_collective_ops(x):
      return [
          ipu.ops.all_to_all_op.all_gather(
              x, replication_factor=replica_group_size)
          for replica_group_size in [1, 2, 4]
      ]

    inputs = np.array(
        [
            [0., 0., 1., 2.],  # replica 0
            [0., 1., 2., 3.],  # replica 1
            [1., 2., 3., 4.],  # replica 2
            [2., 3., 4., 5.],  # replica 3
        ],
        dtype=np.float32)

    outputs = run_collective_ops(inputs, my_collective_ops, num_replicas=4)
    [group_size_1], [group_size_2], [group_size_4] = outputs

    # Group size 1.
    self.assertAllEqual(group_size_1[0], inputs[0:1])
    self.assertAllEqual(group_size_1[1], inputs[1:2])
    self.assertAllEqual(group_size_1[2], inputs[2:3])
    self.assertAllEqual(group_size_1[3], inputs[3:4])

    # Group size 2.
    self.assertAllEqual(group_size_2[0], inputs[0:2])
    self.assertAllEqual(group_size_2[1], inputs[0:2])
    self.assertAllEqual(group_size_2[2], inputs[2:4])
    self.assertAllEqual(group_size_2[3], inputs[2:4])

    # Group size 4.
    self.assertAllEqual(group_size_4[0], inputs)
    self.assertAllEqual(group_size_4[1], inputs)
    self.assertAllEqual(group_size_4[2], inputs)
    self.assertAllEqual(group_size_4[3], inputs)

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def test_grouped_all_reduce(self):
    def my_collective_ops(x):
      return [
          ipu.ops.cross_replica_ops.cross_replica_sum(
              x, replica_group_size=replica_group_size)
          for replica_group_size in [1, 2, 4]
      ]

    inputs = np.array(
        [
            [0., 0., 1., 2.],  # replica 0
            [0., 1., 2., 3.],  # replica 1
            [1., 2., 3., 4.],  # replica 2
            [2., 3., 4., 5.],  # replica 3
        ],
        dtype=np.float32)

    outputs = run_collective_ops(inputs, my_collective_ops, num_replicas=4)
    [group_size_1], [group_size_2], [group_size_4] = outputs

    # Group size 1.
    self.assertAllEqual(group_size_1[0], inputs[0])
    self.assertAllEqual(group_size_1[1], inputs[1])
    self.assertAllEqual(group_size_1[2], inputs[2])
    self.assertAllEqual(group_size_1[3], inputs[3])

    # Group size 2.
    self.assertAllEqual(group_size_2[0], inputs[0:2].sum(axis=0))
    self.assertAllEqual(group_size_2[1], inputs[0:2].sum(axis=0))
    self.assertAllEqual(group_size_2[2], inputs[2:4].sum(axis=0))
    self.assertAllEqual(group_size_2[3], inputs[2:4].sum(axis=0))

    # Group size 4.
    self.assertAllEqual(group_size_4[0], inputs[0:4].sum(axis=0))
    self.assertAllEqual(group_size_4[1], inputs[0:4].sum(axis=0))
    self.assertAllEqual(group_size_4[2], inputs[0:4].sum(axis=0))
    self.assertAllEqual(group_size_4[3], inputs[0:4].sum(axis=0))

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def test_invalid_group_size(self):
    def my_collective_ops(x):
      return ipu.ops.cross_replica_ops.cross_replica_sum(x,
                                                         replica_group_size=3)

    inputs = np.eye(4, dtype=np.float32)

    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        r"The replica group size \(got 3\) does not evenly divide the "
        r"total number of replicas \(got 4\)"):
      run_collective_ops(inputs, my_collective_ops, num_replicas=4)


if __name__ == "__main__":
  googletest.main()
