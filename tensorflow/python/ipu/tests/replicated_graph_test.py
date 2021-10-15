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
import six
import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.client import session as sl
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import googletest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables


class TestReplicatedGraph(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testCreateSimpleReplicatedGraph(self):
    def my_graph(inp):
      with ops.device("/device:IPU:0"):
        x = inp + inp

        return [ipu.ops.cross_replica_ops.cross_replica_sum(x)]

    with ops.device('cpu'):
      inp = array_ops.placeholder(np.float32, [4], name="data")

    out = ipu.ipu_compiler.compile(my_graph, [inp])

    cfg = ipu.config.IPUConfig()
    cfg.optimizations.maximum_cross_replica_sum_buffer_size = 10000
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    with sl.Session() as sess:
      sess.run(variables.global_variables_initializer())

      data = np.ones([4])
      fd = {inp: data}

      result = sess.run(out, fd)

      # Test that the output is just the input
      self.assertAllClose(result[0], 4 * data)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testReplicatedGraphWithLossAndGrad(self):
    def my_graph(inp):
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          b = variable_scope.get_variable(
              "b",
              dtype=np.float32,
              shape=[4],
              initializer=init_ops.constant_initializer(3))
        x = inp + b
        loss = ipu.ops.cross_replica_ops.cross_replica_sum(x)
        b_grad = gradients_impl.gradients(loss, b)
        return [loss, b_grad]

    with ops.device('cpu'):
      inp = array_ops.placeholder(np.float32, [4], name="data")

    out = ipu.ipu_compiler.compile(my_graph, [inp])

    cfg = ipu.config.IPUConfig()
    cfg.optimizations.maximum_cross_replica_sum_buffer_size = 10000
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    with sl.Session() as sess:
      sess.run(variables.global_variables_initializer())

      data = np.ones([4])
      fd = {inp: data}

      loss, b_grad = sess.run(out, fd)

      self.assertAllClose(loss, [8, 8, 8, 8])
      self.assertAllClose(b_grad, [[2, 2, 2, 2]])

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testCrossReplicaSumDifferentTypes(self):
    def my_graph(x, y):
      with ops.device("/device:IPU:0"):
        x = x + x
        y = y + y + 1
        return [
            ipu.ops.cross_replica_ops.cross_replica_sum(x),
            ipu.ops.cross_replica_ops.cross_replica_sum(y)
        ]

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, [4], name="data")
      y = array_ops.placeholder(np.int32, [4], name="data")

    out = ipu.ipu_compiler.compile(my_graph, [x, y])

    cfg = ipu.config.IPUConfig()
    cfg.optimizations.maximum_cross_replica_sum_buffer_size = 10000
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    with sl.Session() as sess:
      sess.run(variables.global_variables_initializer())

      ones = np.ones([4])
      fd = {x: ones, y: ones}

      result = sess.run(out, fd)

      # Test that the output is just the input
      self.assertAllClose(result[0], 4 * ones)
      self.assertAllClose(result[1], 6 * ones)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testCreateSimpleReplicatedGraphVariable(self):
    def my_graph():
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          x = variable_scope.get_variable(
              "x",
              dtype=np.float32,
              shape=[4],
              initializer=init_ops.constant_initializer(10.0))
        x = x + x
        return [ipu.ops.cross_replica_ops.cross_replica_sum(x)]

    out = ipu.ipu_compiler.compile(my_graph, [])

    cfg = ipu.config.IPUConfig()
    cfg.optimizations.maximum_cross_replica_sum_buffer_size = 10000
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    with sl.Session() as sess:
      sess.run(variables.global_variables_initializer())

      result = sess.run(out, {})

      # Test that the output is just the input
      self.assertAllClose(result[0], 4 * np.full([4], 10.0))

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testCreateSimpleReplicatedInfeedOutfeed(self):
    shape = [2]
    dataset = tu.create_single_increasing_dataset(3, shape)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def body(v, x):
      v = ipu.ops.cross_replica_ops.cross_replica_sum(v + x)
      outfeed = outfeed_queue.enqueue(v)
      return (v, outfeed)

    def my_net():
      v = constant_op.constant(0.0, shape=shape, dtype=np.float32)
      r = ipu.loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    outfed = outfeed_queue.dequeue()

    cfg = ipu.config.IPUConfig()
    cfg.optimizations.maximum_cross_replica_sum_buffer_size = 10000
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    with sl.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(48, shape))
      outfed_result = sess.run(outfed)

      self.assertTrue(outfed_result.shape[0], 2)
      self.assertAllClose(outfed_result[0][0], outfed_result[0][1])
      self.assertAllClose(outfed_result[0][0], np.broadcast_to(1, shape))

      self.assertAllClose(outfed_result[1][0], outfed_result[1][1])
      self.assertAllClose(outfed_result[1][0], np.broadcast_to(4, shape))

      self.assertAllClose(outfed_result[2][0], outfed_result[2][1])
      self.assertAllClose(outfed_result[2][0], np.broadcast_to(11, shape))

      self.assertAllClose(outfed_result[3][0], outfed_result[3][1])
      self.assertAllClose(outfed_result[3][0], np.broadcast_to(23, shape))

      self.assertAllClose(outfed_result[4][0], outfed_result[4][1])
      self.assertAllClose(outfed_result[4][0], np.broadcast_to(48, shape))

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testCreateSimpleReplicatedInfeedOutfeedTuple(self):
    shape = [2]
    dataset = tu.create_single_increasing_dataset(3, shape)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def body(v, x):
      out = ipu.ops.cross_replica_ops.cross_replica_sum(v + x)
      outfeed = outfeed_queue.enqueue((v, out))
      return (out, outfeed)

    def my_net():
      v = constant_op.constant(0.0, shape=shape, dtype=np.float32)
      r = ipu.loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    outfed = outfeed_queue.dequeue()

    cfg = ipu.config.IPUConfig()
    cfg.optimizations.maximum_cross_replica_sum_buffer_size = 10000
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    with sl.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(48, shape))
      outfed_result = sess.run(outfed)

      self.assertTrue(outfed_result[0].shape[0], 2)
      self.assertTrue(outfed_result[1].shape[0], 2)
      self.assertAllClose(outfed_result[0][0][0], outfed_result[0][0][1])
      self.assertAllClose(outfed_result[0][0][0], np.broadcast_to(0, shape))
      self.assertAllClose(outfed_result[1][0][0], outfed_result[1][0][1])
      self.assertAllClose(outfed_result[1][0][0], np.broadcast_to(1, shape))

      self.assertAllClose(outfed_result[0][1][0], outfed_result[0][1][1])
      self.assertAllClose(outfed_result[0][1][0], np.broadcast_to(1, shape))
      self.assertAllClose(outfed_result[1][1][0], outfed_result[1][1][1])
      self.assertAllClose(outfed_result[1][1][0], np.broadcast_to(4, shape))

      self.assertAllClose(outfed_result[0][2][0], outfed_result[0][2][1])
      self.assertAllClose(outfed_result[0][2][0], np.broadcast_to(4, shape))
      self.assertAllClose(outfed_result[1][2][0], outfed_result[1][2][1])
      self.assertAllClose(outfed_result[1][2][0], np.broadcast_to(11, shape))

      self.assertAllClose(outfed_result[0][3][0], outfed_result[0][3][1])
      self.assertAllClose(outfed_result[0][3][0], np.broadcast_to(11, shape))
      self.assertAllClose(outfed_result[1][3][0], outfed_result[1][3][1])
      self.assertAllClose(outfed_result[1][3][0], np.broadcast_to(23, shape))

      self.assertAllClose(outfed_result[0][4][0], outfed_result[0][4][1])
      self.assertAllClose(outfed_result[0][4][0], np.broadcast_to(23, shape))
      self.assertAllClose(outfed_result[1][4][0], outfed_result[1][4][1])
      self.assertAllClose(outfed_result[1][4][0], np.broadcast_to(48, shape))

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testCreateSimpleReplicatedInfeedOutfeedDict(self):
    shape = [2]
    dataset = tu.create_single_increasing_dataset(3, shape)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def body(v, x):
      out = ipu.ops.cross_replica_ops.cross_replica_sum(v + x)
      outfeed = outfeed_queue.enqueue({"last": v, "this": out})
      return (out, outfeed)

    def my_net():
      v = constant_op.constant(0.0, shape=shape, dtype=np.float32)
      r = ipu.loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    outfed = outfeed_queue.dequeue()

    cfg = ipu.config.IPUConfig()
    cfg.optimizations.maximum_cross_replica_sum_buffer_size = 10000
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    with sl.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(48, shape))
      outfed_result = sess.run(outfed)

      self.assertTrue(outfed_result["last"].shape[0], 2)
      self.assertTrue(outfed_result["this"].shape[0], 2)
      self.assertAllClose(outfed_result["last"][0][0],
                          outfed_result["last"][0][1])
      self.assertAllClose(outfed_result["last"][0][0],
                          np.broadcast_to(0, shape))
      self.assertAllClose(outfed_result["this"][0][0],
                          outfed_result["this"][0][1])
      self.assertAllClose(outfed_result["this"][0][0],
                          np.broadcast_to(1, shape))

      self.assertAllClose(outfed_result["last"][1][0],
                          outfed_result["last"][1][1])
      self.assertAllClose(outfed_result["last"][1][0],
                          np.broadcast_to(1, shape))
      self.assertAllClose(outfed_result["this"][1][0],
                          outfed_result["this"][1][1])
      self.assertAllClose(outfed_result["this"][1][0],
                          np.broadcast_to(4, shape))

      self.assertAllClose(outfed_result["last"][2][0],
                          outfed_result["last"][2][1])
      self.assertAllClose(outfed_result["last"][2][0],
                          np.broadcast_to(4, shape))
      self.assertAllClose(outfed_result["this"][2][0],
                          outfed_result["this"][2][1])
      self.assertAllClose(outfed_result["this"][2][0],
                          np.broadcast_to(11, shape))

      self.assertAllClose(outfed_result["last"][3][0],
                          outfed_result["last"][3][1])
      self.assertAllClose(outfed_result["last"][3][0],
                          np.broadcast_to(11, shape))
      self.assertAllClose(outfed_result["this"][3][0],
                          outfed_result["this"][3][1])
      self.assertAllClose(outfed_result["this"][3][0],
                          np.broadcast_to(23, shape))

      self.assertAllClose(outfed_result["last"][4][0],
                          outfed_result["last"][4][1])
      self.assertAllClose(outfed_result["last"][4][0],
                          np.broadcast_to(23, shape))
      self.assertAllClose(outfed_result["this"][4][0],
                          outfed_result["this"][4][1])
      self.assertAllClose(outfed_result["this"][4][0],
                          np.broadcast_to(48, shape))

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testCreateCombinedReplicatedSumGraph(self):
    def my_graph():
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          x1 = variable_scope.get_variable(
              "x1",
              dtype=np.float32,
              shape=[100],
              initializer=init_ops.constant_initializer(10.0))
          x2 = variable_scope.get_variable(
              "x2",
              dtype=np.int32,
              shape=[100],
              initializer=init_ops.constant_initializer(10))
        y1 = ipu.ops.cross_replica_ops.cross_replica_sum(x1 + x1)
        z1 = ipu.ops.cross_replica_ops.cross_replica_sum(x1 * x1)
        y2 = ipu.ops.cross_replica_ops.cross_replica_sum(x2 + x2)
        z2 = ipu.ops.cross_replica_ops.cross_replica_sum(x2 * x2)
        return [
            ipu.ops.cross_replica_ops.cross_replica_sum(z1 + y1),
            ipu.ops.cross_replica_ops.cross_replica_sum(z2 + y2)
        ]

    out = ipu.ipu_compiler.compile(my_graph, [])
    cfg = ipu.config.IPUConfig()
    cfg.optimizations.maximum_cross_replica_sum_buffer_size = 10000
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    with sl.Session() as sess:
      sess.run(variables.global_variables_initializer())

      result = sess.run(out, {})
      ref = np.empty([2, 100])
      ref.fill(480.0)

      # Check output equals the expected value
      self.assertAllClose(result, ref)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testReplicatedGraphWithoutAllReduce(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3, 4])

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def body(x):
      outfeed = outfeed_queue.enqueue(x)
      return outfeed

    def my_net():
      r = ipu.loops.repeat(2, body, infeed_queue=infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net)

    outfed = outfeed_queue.dequeue()

    cfg = ipu.config.IPUConfig()
    cfg.optimizations.maximum_cross_replica_sum_buffer_size = 10000
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    with sl.Session() as sess:
      sess.run(infeed_queue.initializer)
      sess.run(res)
      outfed_result = sess.run(outfed)

    self.assertAllClose([[1, 2], [3, 4]], outfed_result)


if __name__ == "__main__":
  googletest.main()
