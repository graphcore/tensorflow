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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import test_utils as tu

from tensorflow.compiler.plugin.poplar.ops import gen_sendrecv_ops
from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


def next_feed_id():
  result = 'feed' + str(next_feed_id.feed_count)
  next_feed_id.feed_count += 1
  return result


next_feed_id.feed_count = 0


def _configure_replicated_ipu_system():
  cfg = ipu.utils.create_ipu_config(profiling=True)
  cfg = ipu.utils.set_optimization_options(
      cfg,
      max_cross_replica_sum_buffer_size=10000,
      max_reduce_scatter_buffer_size=10000)
  cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
  cfg = ipu.utils.auto_select_ipus(cfg, 2)
  ipu.utils.configure_ipu_system(cfg)


class ReplicatedGraphTest(xla_test.XLATestCase):
  def testCreateSimpleReplicatedGraph(self):
    with self.session() as sess:

      def my_graph(inp):
        with ops.device("/device:IPU:0"):
          x = inp + inp

          return [ipu.ops.cross_replica_ops.cross_replica_sum(x)]

      with ops.device('cpu'):
        inp = array_ops.placeholder(np.float32, [4], name="data")

      out = ipu.ipu_compiler.compile(my_graph, [inp])

      _configure_replicated_ipu_system()

      sess.run(variables.global_variables_initializer())

      data = np.ones([4])
      fd = {inp: data}

      result = sess.run(out, fd)

      # Test that the output is just the input
      self.assertAllClose(result[0], 4 * data)

  def testCrossReplicaSumDifferentTypes(self):
    with self.session() as sess:

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

      _configure_replicated_ipu_system()

      sess.run(variables.global_variables_initializer())

      ones = np.ones([4])
      fd = {x: ones, y: ones}

      result = sess.run(out, fd)

      # Test that the output is just the input
      self.assertAllClose(result[0], 4 * ones)
      self.assertAllClose(result[1], 6 * ones)

  def testCreateSimpleReplicatedGraphVariable(self):
    with self.session() as sess:

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

      _configure_replicated_ipu_system()

      sess.run(variables.global_variables_initializer())

      result = sess.run(out, {})

      # Test that the output is just the input
      self.assertAllClose(result[0], 4 * np.full([4], 10.0))

  def testCreateSimpleReplicatedInfeedOutfeed(self):
    with self.session() as sess:
      shape = [2]
      dataset = tu.create_single_increasing_dataset(3, shape)

      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(
          dataset, feed_name=next_feed_id(), replication_factor=2)
      outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
          feed_name=next_feed_id(), replication_factor=2)

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

      _configure_replicated_ipu_system()

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

  def testCreateSimpleReplicatedInfeedOutfeedTuple(self):
    with self.session() as sess:
      shape = [2]
      dataset = tu.create_single_increasing_dataset(3, shape)

      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(
          dataset, feed_name=next_feed_id(), replication_factor=2)
      outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
          feed_name=next_feed_id(), replication_factor=2)

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

      _configure_replicated_ipu_system()

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

  def testCreateSimpleReplicatedInfeedOutfeedDict(self):
    with self.session() as sess:
      shape = [2]
      dataset = tu.create_single_increasing_dataset(3, shape)

      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(
          dataset, feed_name=next_feed_id(), replication_factor=2)
      outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
          feed_name=next_feed_id(), replication_factor=2)

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

      _configure_replicated_ipu_system()

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

  def testCreateCombinedReplicatedSumGraph(self):
    with self.session() as sess:

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

      _configure_replicated_ipu_system()

      sess.run(variables.global_variables_initializer())

      result = sess.run(out, {})
      ref = np.empty([2, 100])
      ref.fill(480.0)

      # Check output equals the expected value
      self.assertAllClose(result, ref)

  def testReplicatedGraphWithoutAllReduce(self):
    with self.session() as sess:
      dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3, 4])

      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(
          dataset, feed_name=next_feed_id(), replication_factor=2)
      outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
          feed_name=next_feed_id(), replication_factor=2)

      def body(x):
        outfeed = outfeed_queue.enqueue(x)
        return outfeed

      def my_net():
        r = ipu.loops.repeat(2, body, infeed_queue=infeed_queue)
        return r

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(my_net)

      outfed = outfeed_queue.dequeue()

      _configure_replicated_ipu_system()

      sess.run(infeed_queue.initializer)
      sess.run(res)
      outfed_result = sess.run(outfed)

    self.assertAllClose([[1, 2], [3, 4]], outfed_result)

  def testCreateSimpleReplicatedInfeedWrongReplicationFactor(self):
    with self.session() as sess:
      shape = [2]
      dataset = tu.create_single_increasing_dataset(3, shape)

      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(
          dataset, feed_name=next_feed_id(), replication_factor=4)

      def body(v, x):
        v = ipu.ops.cross_replica_ops.cross_replica_sum(v + x)
        return v

      def my_net():
        v = constant_op.constant(0.0, shape=shape, dtype=np.float32)
        r = ipu.loops.repeat(5, body, [v], infeed_queue)
        return r

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(my_net, inputs=[])

      _configure_replicated_ipu_system()

      sess.run(infeed_queue.initializer)
      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          'Current program has been created with replication_factor 2'):
        sess.run(res)

  def testCreateSimpleReplicatedOutfeedWrongReplicationFactor(self):
    with self.session() as sess:
      shape = [2]

      outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
          feed_name=next_feed_id(), replication_factor=4)

      def body(v):
        v = ipu.ops.cross_replica_ops.cross_replica_sum(v)
        outfeed = outfeed_queue.enqueue(v)
        return (v, outfeed)

      def my_net():
        v = constant_op.constant(0.0, shape=shape, dtype=np.float32)
        r = ipu.loops.repeat(5, body, [v])
        return r

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(my_net, inputs=[])

      _configure_replicated_ipu_system()

      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          'Current program has been created with replication_factor 2'):
        sess.run(res)

  def testReplicatedGraphWithOutsideCompilationScope(self):
    with self.session() as sess:

      def my_net():
        with ipu.scopes.ipu_scope("/device:IPU:0"):
          x = ipu.replication_ops.replication_index()
          with ipu.scopes.outside_compilation_scope():
            # This receives the data from the first replica,
            # and then broadcasts the result to all replicas.
            # So both replicas should receive 0 + 1 = 1 from
            # the host computation.
            x += 1
          return ipu.ops.cross_replica_ops.cross_replica_sum(x)

      [res] = ipu.ipu_compiler.compile(my_net, inputs=[])

      _configure_replicated_ipu_system()

      # Both replicas should receive 1.
      self.assertEqual(2, sess.run(res))

  def testReplicatedReduceScatter(self):

    with self.session() as sess:

      replication_factor = 2

      outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
          feed_name=next_feed_id(), replication_factor=replication_factor)

      def my_net(x):
        with self.assertRaisesRegex(ValueError,
                                    "Shape must be rank 1 but is rank 2"):
          ipu.ops.reduce_scatter_op.reduce_scatter(
              [x], replication_factor=replication_factor)

        y = ipu.ops.reduce_scatter_op.reduce_scatter(
            x, replication_factor=replication_factor)

        self.assertEqual(1, len(y.shape))
        expected_length = np.ceil(int(x.shape[0]) / replication_factor)
        self.assertEqual(expected_length, y.shape[0])

        return outfeed_queue.enqueue(y)

      num_elements = 5  # To test padding

      inputs = [np.arange(num_elements, dtype=np.float32)]
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        compiled_net = ipu.ipu_compiler.compile(my_net, inputs=inputs)

      with ops.device("/device:CPU:0"):
        scattered_chunks = outfeed_queue.dequeue()
        gathered_padded = array_ops.reshape(scattered_chunks, shape=[-1])
        gathered = array_ops.slice(gathered_padded, [0], [num_elements])

      _configure_replicated_ipu_system()

      sess.run(compiled_net)
      gathered_result = sess.run(gathered)
      expected_result = replication_factor * np.arange(num_elements)
      self.assertAllEqual(expected_result, gathered_result)

  def testReplicatedReduceScatterCombining(self):

    with self.session() as sess:

      num_replicas = 2

      outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
          feed_name=next_feed_id(), replication_factor=num_replicas)

      def my_net(*xs):
        y = [
            ipu.ops.reduce_scatter_op.reduce_scatter(
                x, replication_factor=num_replicas) for x in xs
        ]
        return outfeed_queue.enqueue(y)

      inputs = [i * np.arange(i, dtype=np.float32) for i in range(1, 6)]
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        compiled_net = ipu.ipu_compiler.compile(my_net, inputs=inputs)

      gathered = []
      with ops.device("/device:CPU:0"):
        dequeued = outfeed_queue.dequeue()
        for scattered in dequeued:
          gathered.append(array_ops.reshape(scattered, shape=[-1]))

      _configure_replicated_ipu_system()

      report = tu.ReportJSON(self, sess, configure_device=False)
      report.reset()

      sess.run(compiled_net)
      out = sess.run(gathered)

      # Check that the reduce scatters were combined into one.
      report.parse_log()
      report.assert_compute_sets_matches(
          "IpuReduceScatter*/reduce-scatter*/ReduceScatter", 1)

      # Check padded lengths.
      self.assertEqual(len(out[0]), np.ceil(1 / num_replicas) * num_replicas)
      self.assertEqual(len(out[1]), np.ceil(2 / num_replicas) * num_replicas)
      self.assertEqual(len(out[2]), np.ceil(3 / num_replicas) * num_replicas)
      self.assertEqual(len(out[3]), np.ceil(4 / num_replicas) * num_replicas)
      self.assertEqual(len(out[4]), np.ceil(5 / num_replicas) * num_replicas)

      # Check payloads.
      self.assertAllEqual(1.0 * num_replicas * np.arange(1), out[0][:1])
      self.assertAllEqual(2.0 * num_replicas * np.arange(2), out[1][:2])
      self.assertAllEqual(3.0 * num_replicas * np.arange(3), out[2][:3])
      self.assertAllEqual(4.0 * num_replicas * np.arange(4), out[3][:4])
      self.assertAllEqual(5.0 * num_replicas * np.arange(5), out[4][:5])

  def testSendToHostConcat(self):
    with self.session() as sess:

      def device_fn():
        x = [ipu.ops.replication_ops.replication_index()]
        return gen_sendrecv_ops.ipu_send_to_host(
            x,
            tensor_name="replication_index",
            send_device="/device:IPU:0",
            send_device_incarnation=0,
            recv_device="/device:CPU:0",
            replica_handling="Concat")

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        send_op = ipu.ipu_compiler.compile(device_fn, inputs=[])

      num_replicas = 2

      with ops.device("/device:CPU:0"):
        recv_op = gen_sendrecv_ops.ipu_recv_at_host(
            T=np.int32,
            tensor_name="replication_index",
            send_device="/device:IPU:0",
            send_device_incarnation=0,
            recv_device="/device:CPU:0")

      _configure_replicated_ipu_system()

      # Test a couple of times to ensure the communication can be repeated.
      for _ in range(2):
        _, received = sess.run([send_op, recv_op])
        self.assertEqual(num_replicas, len(received))
        self.assertEqual(0, received[0])
        self.assertEqual(1, received[1])


if __name__ == "__main__":
  googletest.main()
