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

import os
import numpy as np

from tensorflow.compiler.plugin.poplar.ops import gen_pop_datastream_ops
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


def next_feed_id():
  result = 'feed' + str(next_feed_id.feed_count)
  next_feed_id.feed_count += 1
  return result


next_feed_id.feed_count = 0


class PopDatastreamTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testSingleOutfeed(self):
    feed_name = next_feed_id()
    shape = [10, 10]
    with ops.device("/device:IPU:0"):
      a = array_ops.placeholder(np.float32, shape)
      b = array_ops.placeholder(np.float32, shape)
      add = math_ops.add(a, b)
      outfeed_op = gen_pop_datastream_ops.pop_datastream_outfeed_enqueue(
          [add],
          feed_id=feed_name,
          replication_factor=1,
          io_batch_size=1,
          output_shapes=[shape])

    with ops.device('cpu'):
      outfeed = gen_pop_datastream_ops.pop_datastream_outfeed_dequeue(
          feed_id=feed_name,
          replication_factor=1,
          output_types=[np.float32],
          output_shapes=[shape])

    with session_lib.Session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(outfeed_op,
               feed_dict={
                   a: np.ones(shape, np.float32),
                   b: np.ones(shape, np.float32)
               })
      outfed = sess.run(outfeed)
      self.assertEqual(len(outfed[0]), 1)
      self.assertAllClose(outfed[0][0], 2 * np.ones(shape, np.float32))

  @test_util.deprecated_graph_mode_only
  def testTupleOutfeedGetAll(self):
    feed_name = next_feed_id()
    shape_1 = [10, 10]
    shape_2 = [4, 4]

    with ops.device("/device:IPU:0"):
      a = array_ops.placeholder(np.float32, shape_1)
      b = array_ops.placeholder(np.float32, shape_1)
      c = array_ops.placeholder(np.float32, shape_2)
      d = array_ops.placeholder(np.float32, shape_2)
      add = math_ops.add(a, b)
      sub = math_ops.sub(c, d)
      outfeed_op = gen_pop_datastream_ops.pop_datastream_outfeed_enqueue(
          [add, sub],
          feed_id=feed_name,
          replication_factor=1,
          io_batch_size=1,
          output_shapes=[shape_1, shape_2])

    with ops.device('cpu'):
      outfeed = gen_pop_datastream_ops.pop_datastream_outfeed_dequeue(
          feed_id=feed_name,
          replication_factor=1,
          output_types=[np.float32, np.float32],
          output_shapes=[shape_1, shape_2])

    with session_lib.Session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(outfeed_op,
               feed_dict={
                   a: np.ones(shape_1, np.float32),
                   b: np.ones(shape_1, np.float32),
                   c: np.ones(shape_2, np.float32),
                   d: np.ones(shape_2, np.float32)
               })
      sess.run(outfeed_op,
               feed_dict={
                   a: 2 * np.ones(shape_1, np.float32),
                   b: np.ones(shape_1, np.float32),
                   c: 2 * np.ones(shape_2, np.float32),
                   d: np.ones(shape_2, np.float32)
               })
      outfed = sess.run(outfeed)
      self.assertTrue(len(outfed) == 2)
      self.assertEqual(outfed[0].shape, (2, 10, 10))
      self.assertEqual(outfed[1].shape, (2, 4, 4))
      self.assertAllClose(outfed[0][0], np.broadcast_to(2, [10, 10]))
      self.assertAllClose(outfed[0][1], np.broadcast_to(3, [10, 10]))
      self.assertAllClose(outfed[1][0], np.broadcast_to(0, [4, 4]))
      self.assertAllClose(outfed[1][1], np.broadcast_to(1, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testTupleOutfeedGetLast(self):
    feed_name = next_feed_id()
    shape_1 = [10, 10]
    shape_2 = [4, 4]

    with ops.device("/device:IPU:0"):
      a = array_ops.placeholder(np.float32, shape_1)
      b = array_ops.placeholder(np.float32, shape_1)
      c = array_ops.placeholder(np.float32, shape_2)
      d = array_ops.placeholder(np.float32, shape_2)
      add = math_ops.add(a, b)
      sub = math_ops.sub(c, d)
      outfeed_op = gen_pop_datastream_ops.pop_datastream_outfeed_enqueue(
          [add, sub],
          feed_id=feed_name,
          replication_factor=1,
          io_batch_size=1,
          outfeed_mode='get_last',
          output_shapes=[shape_1, shape_2])

    with ops.device('cpu'):
      outfeed = gen_pop_datastream_ops.pop_datastream_outfeed_dequeue(
          feed_id=feed_name,
          replication_factor=1,
          outfeed_mode='get_last',
          output_types=[np.float32, np.float32],
          output_shapes=[shape_1, shape_2])

    with session_lib.Session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(outfeed_op,
               feed_dict={
                   a: np.ones(shape_1, np.float32),
                   b: np.ones(shape_1, np.float32),
                   c: np.ones(shape_2, np.float32),
                   d: np.ones(shape_2, np.float32)
               })
      sess.run(outfeed_op,
               feed_dict={
                   a: 2 * np.ones(shape_1, np.float32),
                   b: np.ones(shape_1, np.float32),
                   c: 2 * np.ones(shape_2, np.float32),
                   d: np.ones(shape_2, np.float32)
               })
      outfed = sess.run(outfeed)
      self.assertTrue(len(outfed) == 2)
      self.assertEqual(outfed[0].shape, (10, 10))
      self.assertEqual(outfed[1].shape, (4, 4))
      self.assertAllClose(outfed[0], np.broadcast_to(3, [10, 10]))
      self.assertAllClose(outfed[1], np.broadcast_to(1, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testOutfeedGetAll(self):
    feed_name = next_feed_id()
    shape = [2, 2]
    with ops.device("/device:IPU:0"):
      a = array_ops.placeholder(np.float32, shape)
      b = array_ops.placeholder(np.float32, shape)
      add = math_ops.add(a, b)
      outfeed_op = gen_pop_datastream_ops.pop_datastream_outfeed_enqueue(
          [add],
          feed_id=feed_name,
          replication_factor=1,
          io_batch_size=1,
          outfeed_mode='all',
          output_shapes=[shape])

    with ops.device('cpu'):
      outfeed_all = gen_pop_datastream_ops.pop_datastream_outfeed_dequeue(
          feed_id=feed_name,
          replication_factor=1,
          output_types=[np.float32],
          output_shapes=[shape])

    with session_lib.Session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(outfeed_op,
               feed_dict={
                   a: np.ones(shape, np.float32),
                   b: np.ones(shape, np.float32)
               })
      sess.run(outfeed_op,
               feed_dict={
                   a: 3.1 * np.ones(shape, np.float32),
                   b: 2 * np.ones(shape, np.float32)
               })

      outfed = sess.run(outfeed_all)
      self.assertTrue(len(outfed) == 1)
      self.assertEqual(outfed[0].shape, (2, 2, 2))
      self.assertAllClose(outfed[0][0], 2 * np.ones(shape, np.float32))
      self.assertAllClose(outfed[0][1], (3.1 + 2) * np.ones(shape, np.float32))

  @test_util.deprecated_graph_mode_only
  def testOutfeedGetLast(self):
    feed_name = next_feed_id()
    shape = [2, 2]
    with ops.device("/device:IPU:0"):
      a = array_ops.placeholder(np.float32, shape)
      b = array_ops.placeholder(np.float32, shape)
      add = math_ops.add(a, b)
      outfeed_op = gen_pop_datastream_ops.pop_datastream_outfeed_enqueue(
          [add],
          feed_id=feed_name,
          replication_factor=1,
          io_batch_size=1,
          outfeed_mode='get_last',
          output_shapes=[shape])

    with ops.device('cpu'):
      outfeed_last = gen_pop_datastream_ops.pop_datastream_outfeed_dequeue(
          feed_id=feed_name,
          replication_factor=1,
          outfeed_mode='get_last',
          output_types=[np.float32],
          output_shapes=[shape])

    with session_lib.Session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(outfeed_op,
               feed_dict={
                   a: np.ones(shape, np.float32),
                   b: np.ones(shape, np.float32)
               })
      sess.run(outfeed_op,
               feed_dict={
                   a: 3.1 * np.ones(shape, np.float32),
                   b: 2 * np.ones(shape, np.float32)
               })

      outfed = sess.run(outfeed_last)
      self.assertTrue(len(outfed) == 1)
      self.assertEqual(outfed[0].shape, (2, 2))
      self.assertAllClose(outfed[0], (3.1 + 2) * np.ones(shape, np.float32))


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
