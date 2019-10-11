# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent


def next_feed_id():
  result = 'feed' + str(next_feed_id.feed_count)
  next_feed_id.feed_count += 1
  return result


next_feed_id.feed_count = 0


class ShardedAndReplicatedTest(xla_test.XLATestCase):
  def testShardedAndReplicated(self):
    with self.session() as sess:
      shape = [2]
      dataset = tu.create_single_increasing_dataset(3, shape)

      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(
          dataset, feed_name=next_feed_id(), replication_factor=2)
      outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
          feed_name=next_feed_id(), replication_factor=2)

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

      tu.ReportJSON(self,
                    sess,
                    execution_trace=False,
                    max_cross_replica_sum_buffer_size=10000,
                    max_inter_ipu_copies_buffer_size=10000,
                    device_count_override=4)

      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(10, shape))
      outfed_result = sess.run(outfed)

      self.assertTrue(outfed_result.shape[0], 2)
      self.assertAllClose(outfed_result[0][0], outfed_result[0][1])
      self.assertAllClose(outfed_result[0][0], np.broadcast_to(2, shape))

      self.assertAllClose(outfed_result[1][0], outfed_result[1][1])
      self.assertAllClose(outfed_result[1][0], np.broadcast_to(10, shape))

  def testShardedAndReplicatedTraining(self):
    with self.session() as sess:

      def my_graph(inp, lab):
        with ops.device("/device:IPU:0"):
          with ipu.scopes.ipu_shard(0):
            x = layers.Conv2D(8, 3, padding='same', name="convA")(inp)

          with ipu.scopes.ipu_shard(1):
            x = layers.Conv2D(8, 1, padding='same', name="convB")(x)
            x = math_ops.reduce_mean(x, axis=[1, 2])

            loss = nn.softmax_cross_entropy_with_logits_v2(
                logits=x, labels=array_ops.stop_gradient(lab))
            loss = math_ops.reduce_mean(loss)

          opt = ipu.ipu_optimizer.CrossReplicaOptimizer(
              ipu.sharded_optimizer.ShardedOptimizer(
                  gradient_descent.GradientDescentOptimizer(0.000001)))
          train = opt.minimize(loss)

        return [loss, train]

      with ops.device('cpu'):
        inp = array_ops.placeholder(np.float32, [1, 32, 32, 4], name="data")
        lab = array_ops.placeholder(np.float32, [1, 8], name="labels")

      out = ipu.ipu_compiler.compile(my_graph, [inp, lab])

      report = tu.ReportJSON(self,
                             sess,
                             execution_trace=False,
                             max_cross_replica_sum_buffer_size=10000,
                             max_inter_ipu_copies_buffer_size=10000,
                             device_count_override=4)

      sess.run(variables.global_variables_initializer())
      report.reset()

      fd = {inp: np.ones([1, 32, 32, 4]), lab: np.ones([1, 8])}
      sess.run(out, fd)

      report.parse_log()
      report.assert_compute_sets_matches(
          "*/GlobalPre/*", 6, "There should be 6 global communications")

  def testShardedAndReplicatedAndGradientAccumulateTraining(self):
    with self.session() as sess:
      dataset = tu.create_dual_increasing_dataset(3)

      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(
          dataset, feed_name=next_feed_id(), replication_factor=2)

      def my_graph(loss, inp, lab):
        with ops.device("/device:IPU:0"):
          with ipu.scopes.ipu_shard(0):
            x = layers.Conv2D(8, 3, padding='same', name="convA")(inp)

          with ipu.scopes.ipu_shard(1):
            x = layers.Conv2D(8, 1, padding='same', name="convB")(x)
            x = math_ops.reduce_mean(x, axis=[1, 2])

            loss = nn.softmax_cross_entropy_with_logits_v2(
                logits=x, labels=array_ops.stop_gradient(lab))
            loss = math_ops.reduce_mean(loss)

          opt = ipu.gradient_accumulation_optimizer.CrossReplicaGradientAccumulationOptimizer(
              ipu.sharded_optimizer.ShardedOptimizer(
                  gradient_descent.GradientDescentOptimizer(0.000001)), 10)
          train = opt.minimize(loss)

        return [loss, train]

      def my_net():
        v = 0.0
        r = ipu.loops.repeat(2, my_graph, [v], infeed_queue)
        return r

      out = ipu.ipu_compiler.compile(my_net, [])

      report = tu.ReportJSON(self,
                             sess,
                             execution_trace=False,
                             max_cross_replica_sum_buffer_size=10000,
                             max_inter_ipu_copies_buffer_size=10000,
                             device_count_override=4)

      sess.run(infeed_queue.initializer)
      sess.run(variables.global_variables_initializer())
      report.reset()

      sess.run(out)

      report.parse_log()
      report.assert_compute_sets_matches(
          "*/GlobalPre/*", 6, "There should be 6 global communications")


if __name__ == "__main__":
  googletest.main()
