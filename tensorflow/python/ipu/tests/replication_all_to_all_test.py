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

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python import ipu
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables


class TestAllGather(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=8)
  @test_util.deprecated_graph_mode_only
  def testAllGather(self):
    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[8])

    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def my_graph(x):
      with ops.device("/device:IPU:0"):
        rep_id = ipu.replication_ops.replication_index()
        x = x + math_ops.cast(rep_id, dtype=np.float32)
        y, = gen_popops_ops.ipu_all_gather([x], replication_factor=8)
        outfeed = outfeed_queue.enqueue(y)
        return y, outfeed

    out = ipu.ipu_compiler.compile(my_graph, [x])

    config = ipu.config.IPUConfig()
    config.auto_select_ipus = [8]
    tu.add_hw_ci_connection_options(config)
    config.configure_ipu_system()

    outfeed = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(out, {x: np.zeros([8])})
      result = sess.run(outfeed)[0]

      for replica in range(0, 8):
        self.assertAllEqual(result[replica][0], [0] * 8)
        self.assertAllEqual(result[replica][1], [1] * 8)
        self.assertAllEqual(result[replica][2], [2] * 8)
        self.assertAllEqual(result[replica][3], [3] * 8)
        self.assertAllEqual(result[replica][4], [4] * 8)
        self.assertAllEqual(result[replica][5], [5] * 8)
        self.assertAllEqual(result[replica][6], [6] * 8)
        self.assertAllEqual(result[replica][7], [7] * 8)

  @test_util.deprecated_graph_mode_only
  def testAllGatherShapeInference(self):
    x = array_ops.placeholder(np.float32, shape=(2, 4))
    y, = gen_popops_ops.ipu_all_gather([x], replication_factor=8)
    self.assertAllEqual((8, 2, 4), y.shape)

  @tu.test_uses_ipus(num_ipus=8)
  @test_util.deprecated_graph_mode_only
  def testSerializedMultiUpdateAdd(self):
    with ops.device('cpu'):
      idx = array_ops.placeholder(np.int32, shape=[16])
      updates = array_ops.placeholder(np.float32, shape=[16, 128])
      scale = array_ops.placeholder(np.float32, shape=[])

    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def my_graph(idx, updates, scale):
      zero = 0.0
      zeros = array_ops.broadcast_to(zero, shape=[1000, 128])
      out = gen_popops_ops.ipu_multi_update_add(zeros,
                                                updates=updates,
                                                indices=idx,
                                                scale=scale)
      out = gen_popops_ops.ipu_cross_replica_sum(out)
      out = gen_poputil_ops.ipu_replication_normalise(out)
      outfeed = outfeed_queue.enqueue(out)
      return out, outfeed

    with ops.device("/device:IPU:0"):
      out = ipu.ipu_compiler.compile(my_graph, [idx, updates, scale])

    config = ipu.config.IPUConfig()
    config.auto_select_ipus = [8]
    tu.add_hw_ci_connection_options(config)
    config.configure_ipu_system()

    outfeed = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(
          out, {
              idx:
              [1, 2, 3, 4, 1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400],
              updates: np.ones(updates.shape),
              scale: 2,
          })
      result = sess.run(outfeed)

      for replica in range(0, 8):
        t = result[0][replica]
        self.assertAllEqual(t[1], np.full([128], 4.0))
        self.assertAllEqual(t[2], np.full([128], 4.0))
        self.assertAllEqual(t[3], np.full([128], 4.0))
        self.assertAllEqual(t[4], np.full([128], 4.0))
        self.assertAllEqual(t[10], np.full([128], 2.0))
        self.assertAllEqual(t[20], np.full([128], 2.0))
        self.assertAllEqual(t[30], np.full([128], 2.0))
        self.assertAllEqual(t[40], np.full([128], 2.0))
        self.assertAllEqual(t[100], np.full([128], 2.0))
        self.assertAllEqual(t[200], np.full([128], 2.0))
        self.assertAllEqual(t[300], np.full([128], 2.0))
        self.assertAllEqual(t[400], np.full([128], 2.0))


class TestAllToAll(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=8)
  @test_util.deprecated_graph_mode_only
  def testAllToAll(self):
    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[8])

    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def my_graph(x):
      with ops.device("/device:IPU:0"):
        rep_id = ipu.replication_ops.replication_index()

        tmp = 8.0 * math_ops.cast(rep_id, dtype=np.float32)
        x = math_ops.cast(x, dtype=np.float32) + tmp
        y = gen_popops_ops.ipu_all_to_all(x, 0, 0, 8)
        outfeed = outfeed_queue.enqueue(y)
        return y, outfeed

    out = ipu.ipu_compiler.compile(my_graph, [x])

    config = ipu.config.IPUConfig()
    config.auto_select_ipus = [8]
    tu.add_hw_ci_connection_options(config)
    config.configure_ipu_system()

    outfeed = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(out, {x: np.arange(1, 9)})

      result = sess.run(outfeed)[0]
      self.assertAllEqual(result[0], [1, 9, 17, 25, 33, 41, 49, 57])
      self.assertAllEqual(result[1], [2, 10, 18, 26, 34, 42, 50, 58])
      self.assertAllEqual(result[2], [3, 11, 19, 27, 35, 43, 51, 59])
      self.assertAllEqual(result[3], [4, 12, 20, 28, 36, 44, 52, 60])
      self.assertAllEqual(result[4], [5, 13, 21, 29, 37, 45, 53, 61])
      self.assertAllEqual(result[5], [6, 14, 22, 30, 38, 46, 54, 62])
      self.assertAllEqual(result[6], [7, 15, 23, 31, 39, 47, 55, 63])
      self.assertAllEqual(result[7], [8, 16, 24, 32, 40, 48, 56, 64])


if __name__ == "__main__":
  googletest.main()
