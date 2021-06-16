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
import copy
import numpy as np
from tensorflow.python.ipu.config import IPUConfig

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.python import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.ipu.optimizers import cross_replica_optimizer
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer
from tensorflow.python.ipu.optimizers import sharded_optimizer
from tensorflow.python.platform import googletest
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent


class TestShardedAndReplicated(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testShardedAndReplicated(self):
    with sl.Session() as sess:
      shape = [2]
      dataset = tu.create_single_increasing_dataset(3, shape)

      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

      def body(v, x):
        with ipu.scopes.ipu_shard(0):
          z = v + x
          y = x * x
        with ipu.scopes.ipu_shard(1):
          z = (ipu.ops.cross_replica_ops.cross_replica_sum(z) +
               ipu.ops.cross_replica_ops.cross_replica_sum(y))
          outfeed = outfeed_queue.enqueue(z)
        return (z, outfeed)

      def my_net():
        v = constant_op.constant(0.0, shape=shape, dtype=np.float32)
        r = ipu.loops.repeat(2, body, [v], infeed_queue)
        return r

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(my_net, inputs=[])

      outfed = outfeed_queue.dequeue()

      cfg = IPUConfig()
      cfg._profiling.profiling = True  # pylint: disable=protected-access
      cfg.optimizations.maximum_cross_replica_sum_buffer_size = 10000
      cfg.optimizations.maximum_inter_ipu_copies_buffer_size = 10000
      cfg.auto_select_ipus = 4
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()

      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(10, shape))
      outfed_result = sess.run(outfed)

      self.assertTrue(outfed_result.shape[0], 2)
      self.assertAllClose(outfed_result[0][0], outfed_result[0][1])
      self.assertAllClose(outfed_result[0][0], np.broadcast_to(2, shape))

      self.assertAllClose(outfed_result[1][0], outfed_result[1][1])
      self.assertAllClose(outfed_result[1][0], np.broadcast_to(10, shape))

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testShardedAndReplicatedTraining(self):
    with sl.Session() as sess:
      dataset = tu.create_dual_increasing_dataset(3)

      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

      def my_graph(in_loss, inp, lab):
        with ops.device("/device:IPU:0"):
          with ipu.scopes.ipu_shard(0):
            x = layers.Conv2D(8,
                              3,
                              padding='same',
                              use_bias=False,
                              kernel_initializer=init_ops.ones_initializer(),
                              name="convA")(inp)

          with ipu.scopes.ipu_shard(1):
            x = layers.Conv2D(8,
                              1,
                              padding='same',
                              use_bias=False,
                              kernel_initializer=init_ops.ones_initializer(),
                              name="convA")(x)
            x = math_ops.reduce_mean(x, axis=[1, 2])

            loss = nn.softmax_cross_entropy_with_logits_v2(
                logits=x, labels=array_ops.stop_gradient(lab))
            loss = math_ops.reduce_mean(loss)

          opt = cross_replica_optimizer.CrossReplicaOptimizer(
              sharded_optimizer.ShardedOptimizer(
                  gradient_descent.GradientDescentOptimizer(0.000001)))
          train = opt.minimize(loss)
          loss = in_loss + ipu.ops.cross_replica_ops.cross_replica_sum(loss)
        return [loss, train]

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

      def my_net():
        v = 0.0
        r = ipu.loops.repeat(2, my_graph, [v], infeed_queue)
        return r

      out = ipu.ipu_compiler.compile(my_net, [])

      cfg = IPUConfig()
      cfg._profiling.profiling = True  # pylint: disable=protected-access
      cfg.optimizations.maximum_cross_replica_sum_buffer_size = 10000
      cfg.optimizations.maximum_inter_ipu_copies_buffer_size = 10000
      cfg.auto_select_ipus = 4
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()

      sess.run(infeed_queue.initializer)
      sess.run(variables.global_variables_initializer())
      sess.run(report)

      loss = sess.run(out)
      self.assertAllClose(loss, [49.906597])

      rep = sess.run(report)

      num_compiles = 0

      evts = ipu.utils.extract_all_events(rep)
      for evt in evts:
        if evt.type == IpuTraceEvent.COMPILE_END:
          num_compiles = num_compiles + 1

      self.assertEqual(num_compiles, 1)

      compile_report = ipu.utils.extract_compile_reports(rep)
      self.assertEqual(len(compile_report), 1)

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testShardedAndReplicatedAndGradientAccumulateTraining(self):
    with sl.Session() as sess:
      dataset = tu.create_dual_increasing_dataset(3)

      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

      def my_graph(in_loss, inp, lab):
        with ops.device("/device:IPU:0"):
          with ipu.scopes.ipu_shard(0):
            x = layers.Conv2D(8,
                              3,
                              padding='same',
                              use_bias=False,
                              kernel_initializer=init_ops.ones_initializer(),
                              name="convA")(inp)

          with ipu.scopes.ipu_shard(1):
            x = layers.Conv2D(8,
                              1,
                              padding='same',
                              use_bias=False,
                              kernel_initializer=init_ops.ones_initializer(),
                              name="convA")(x)
            x = math_ops.reduce_mean(x, axis=[1, 2])

            loss = nn.softmax_cross_entropy_with_logits_v2(
                logits=x, labels=array_ops.stop_gradient(lab))
            loss = math_ops.reduce_mean(loss)

          opt = gradient_accumulation_optimizer.CrossReplicaGradientAccumulationOptimizer(  # pylint: disable=line-too-long
              sharded_optimizer.ShardedOptimizer(
                  gradient_descent.GradientDescentOptimizer(0.000001)), 10)
          train = opt.minimize(loss)
          loss = in_loss + ipu.ops.cross_replica_ops.cross_replica_sum(loss)
        return [loss, train]

      def my_net():
        v = 0.0
        r = ipu.loops.repeat(20, my_graph, [v], infeed_queue)
        return r

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

      out = ipu.ipu_compiler.compile(my_net, [])

      cfg = IPUConfig()
      cfg._profiling.profiling = True  # pylint: disable=protected-access
      cfg.optimizations.maximum_cross_replica_sum_buffer_size = 10000
      cfg.optimizations.maximum_inter_ipu_copies_buffer_size = 10000
      cfg.auto_select_ipus = 4
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()

      sess.run(infeed_queue.initializer)
      sess.run(variables.global_variables_initializer())
      sess.run(report)

      loss = sess.run(out)
      self.assertAllClose(loss, [648.78577])

      rep = sess.run(report)

      num_compiles = 0

      evts = ipu.utils.extract_all_events(rep)
      for evt in evts:
        if evt.type == IpuTraceEvent.COMPILE_END:
          num_compiles = num_compiles + 1

      self.assertEqual(num_compiles, 1)


@test_util.deprecated_graph_mode_only
class TestMixedShardedAndReplicated(test_util.TensorFlowTestCase):
  @classmethod
  def setUpClass(cls):
    cfg = IPUConfig()
    cfg._profiling.profiling = True  # pylint: disable=protected-access
    cfg.optimizations.maximum_cross_replica_sum_buffer_size = 10000
    cfg.optimizations.maximum_inter_ipu_copies_buffer_size = 10000

    tu.add_hw_ci_connection_options(cfg)
    cls.base_cfg = cfg

  @staticmethod
  def create_body_2shards(outfeed_queue=None):
    def body_sharded(v, x):
      with ipu.scopes.ipu_shard(0):
        z = v + x
        y = x * x
      with ipu.scopes.ipu_shard(1):
        z = (ipu.ops.cross_replica_ops.cross_replica_sum(z) +
             ipu.ops.cross_replica_ops.cross_replica_sum(y))
        if outfeed_queue:
          outfeed = outfeed_queue.enqueue(z)
          return (z, outfeed)
      return z

    return body_sharded

  @staticmethod
  def create_body_not_sharded(outfeed_queue=None):
    def body_not_sharded(v, x):
      v = ipu.ops.cross_replica_ops.cross_replica_sum(v + x)
      if outfeed_queue:
        outfeed = outfeed_queue.enqueue(v)
        return (v, outfeed)
      return v

    return body_not_sharded

  @staticmethod
  def create_network(body, shape, infeed_queue):
    def my_net():
      v = constant_op.constant(0.0, shape=shape, dtype=np.float32)
      r = ipu.loops.repeat(2, body, [v], infeed_queue)
      return r

    return my_net

  @tu.test_uses_ipus(num_ipus=4)
  def testOutfeedShapeWhenMixingReplicationFactors(self):
    ''' Check that we get the correct outfeed shape when running two programs
    with different replication factors'''
    cfg = copy.deepcopy(self.base_cfg)
    cfg.auto_select_ipus = 4
    cfg.configure_ipu_system()

    with sl.Session() as sess:
      shape = [2]
      dataset = tu.create_single_increasing_dataset(3, shape)

      # Setup/run the first program with a replication factor of 2
      infeed_queue_sharded = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue_sharded = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
      body_sharded = self.create_body_2shards(outfeed_queue_sharded)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        result_sharded = ipu.ipu_compiler.compile(self.create_network(
            body_sharded, shape, infeed_queue_sharded),
                                                  inputs=[])

      sess.run(infeed_queue_sharded.initializer)
      sess.run(result_sharded)

      # Setup/run the second program with a replication factor of 4
      infeed_queue_not_sharded = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue_not_sharded = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
      body_not_sharded = self.create_body_not_sharded(
          outfeed_queue_not_sharded)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        result_not_sharded = ipu.ipu_compiler.compile(self.create_network(
            body_not_sharded, shape, infeed_queue_not_sharded),
                                                      inputs=[])

      sess.run(infeed_queue_not_sharded.initializer)
      sess.run(result_not_sharded)

      # Dequeue the outfeeds after running the programs to make sure the the correct replication
      # factor is preserved.
      outfed_result = sess.run(outfeed_queue_sharded.dequeue())
      # The second dimension contains the replication factor
      self.assertEqual(outfed_result.shape[1], 2)

      outfed_result = sess.run(outfeed_queue_not_sharded.dequeue())
      # The second dimension contains the replication factor
      self.assertEqual(outfed_result.shape[1], 4)

  @tu.test_uses_ipus(num_ipus=4)
  def testCantReuseInfeedWhenMixingReplicationFactors(self):

    cfg = copy.deepcopy(self.base_cfg)
    cfg.auto_select_ipus = 4
    cfg.configure_ipu_system()

    with sl.Session() as sess:
      shape = [2]
      dataset = tu.create_single_increasing_dataset(3, shape)

      shared_infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
      sess.run(shared_infeed_queue.initializer)

      # Setup/run the first program with a replication factor of 2
      body_sharded = self.create_body_2shards()
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        result_sharded = ipu.ipu_compiler.compile(self.create_network(
            body_sharded, shape, shared_infeed_queue),
                                                  inputs=[])

      # This setups up shared_infeed_queue with a replication factor of 2
      sess.run(result_sharded)

      # Setup/run the second program with a replication factor of 4
      body_not_sharded = self.create_body_not_sharded()
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        result_not_sharded = ipu.ipu_compiler.compile(self.create_network(
            body_not_sharded, shape, shared_infeed_queue),
                                                      inputs=[])

      # This should error since our infeed queue is already setup with a
      # different replication factor
      self.assertRaises(errors.FailedPreconditionError, sess.run,
                        result_not_sharded)


if __name__ == "__main__":
  googletest.main()
