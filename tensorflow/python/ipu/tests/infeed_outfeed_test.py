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

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent


def next_feed_id():
  result = 'feed' + str(next_feed_id.feed_count)
  next_feed_id.feed_count += 1
  return result


next_feed_id.feed_count = 0


class InfeedOutfeedTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testSingleInfeedRepeatNonTuple(self):
    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())

    def body(v, x):
      v = v + x
      return (v)

    def my_net(v):
      r = ipu.loops.repeat(20, body, (v), infeed_queue)
      return r

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [4, 4])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[v])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res, {v: np.ones([4, 4], np.float32)})
      self.assertAllClose(result[0], np.broadcast_to(91, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedRepeatNonTupleFiniteDataset(self):
    dataset = tu.create_single_increasing_dataset(
        10, shape=[4, 4], repeat=False)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())

    def body(v, x):
      v = v + x
      return (v)

    def my_net(v):
      r = ipu.loops.repeat(10, body, (v), infeed_queue)
      return r

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [4, 4])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[v])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res, {v: np.ones([4, 4], np.float32)})
      self.assertAllClose(result[0], np.broadcast_to(46, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedRepeatTuple(self):
    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())

    def body(v, im1, im2):
      v = v + im1 + im2
      return (v)

    def my_net():
      v = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
      r = ipu.loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(31, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedRepeatTupleMerge(self):
    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())

    def body(v, im1, im2):
      v = v + im1 + im2
      return (v)

    def my_net():
      v = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
      r = ipu.loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    cfg = ipu.utils.create_ipu_config(merge_infeed_io_copies=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(31, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedRepeatNamed(self):
    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return {"a": image_1, "b": image_2}

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())

    # Note how the parameters are swapped around.
    def body(v1, v2, b, a):
      v1 = v1 + a
      v2 = v2 + b
      return (v1, v2)

    def my_net():
      v1 = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
      v2 = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
      r = ipu.loops.repeat(5, body, [v1, v2], infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(4, [4, 4]))
      self.assertAllClose(result[1], np.broadcast_to(27, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedMultipleRepeats(self):
    dataset = tu.create_single_increasing_dataset(2, shape=[4, 4])

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())

    def body(v, x):
      v = v + x
      return (v)

    def my_net():
      v = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
      r = ipu.loops.repeat(5, body, [v], infeed_queue)
      r = ipu.loops.repeat(5, body, [r], infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(5, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedWhileLoopNonTuple(self):
    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())

    def cond(i, v):
      return i < 20

    def body(i, v, x):
      v = v + x
      return (i + 1, v)

    def my_net(v):
      i = 0
      r = ipu.loops.while_loop(cond, body, (i, v), infeed_queue)
      return r[1]

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [4, 4])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[v])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res, {v: np.ones([4, 4], np.float32)})
      self.assertAllClose(result[0], np.broadcast_to(91, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedWhileLoopTuple(self):
    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())

    def cond(i, v):
      return i < 20

    def body(i, v, im1, im2):
      v = v + im1 + im2
      return (i + 1, v)

    def my_net(v):
      i = 0
      r = ipu.loops.while_loop(cond, body, (i, v), infeed_queue)
      return r[1]

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [4, 4])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[v])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res, {v: np.ones([4, 4], np.float32)})
      self.assertAllClose(result[0], np.broadcast_to(129.5, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedMultipleRuns(self):
    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())

    def program(iters):
      def body(v, x):
        v = v + x
        return (v)

      def my_net():
        v = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
        r = ipu.loops.repeat(iters, body, (v), infeed_queue)
        return r

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        return ipu.ipu_compiler.compile(my_net)

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(program(0))
      self.assertAllClose(result[0], np.broadcast_to(0, [4, 4]))
      # The iterator has not moved - next element should be all 1s.
      result = sess.run(program(2))
      self.assertAllClose(result[0], np.broadcast_to(1, [4, 4]))
      # The iterator has moved - in the next two iterations it should pull 2 and 3.
      result = sess.run(program(2))
      self.assertAllClose(result[0], np.broadcast_to(5, [4, 4]))
      # The iterator has moved - in the next two iterations it should pull 4 and 5.
      result = sess.run(program(2))
      self.assertAllClose(result[0], np.broadcast_to(9, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testTwoInfeedsDifferentPrograms(self):
    dataset1 = tu.create_single_increasing_dataset(20, shape=[4, 4])
    dataset2 = tu.create_single_increasing_dataset(3, shape=[4, 4])

    infeed_queue1 = ipu.ipu_infeed_queue.IPUInfeedQueue(
        dataset1, feed_name=next_feed_id())
    infeed_queue2 = ipu.ipu_infeed_queue.IPUInfeedQueue(
        dataset2, feed_name=next_feed_id())

    def program(iters, infeed_queue):
      def body(v, x):
        v = v + x
        return (v)

      def my_net():
        v = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
        r = ipu.loops.repeat(iters, body, (v), infeed_queue)
        return r

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        return ipu.ipu_compiler.compile(my_net)

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue1.initializer)
      sess.run(infeed_queue2.initializer)
      result = sess.run(program(5, infeed_queue1))
      self.assertAllClose(result[0], np.broadcast_to(10, [4, 4]))
      result = sess.run(program(5, infeed_queue2))
      self.assertAllClose(result[0], np.broadcast_to(4, [4, 4]))
      result = sess.run(program(5, infeed_queue1))
      self.assertAllClose(result[0], np.broadcast_to(35, [4, 4]))
      result = sess.run(program(5, infeed_queue2))
      self.assertAllClose(result[0], np.broadcast_to(5, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testUndefinedShape(self):
    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])
    dataset = dataset.batch(10, drop_remainder=False)
    with self.assertRaisesRegexp(ValueError, 'Output shape \((\?|None),'):
      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(
          dataset, next_feed_id())

  @test_util.deprecated_graph_mode_only
  def testMultipleInitializations(self):
    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    infeed_queue.initializer
    with self.assertRaisesRegexp(
        ValueError,
        'The IPUInfeedQueue `initializer` function can only be accessed once.'
    ):
      infeed_queue.initializer

  @test_util.deprecated_graph_mode_only
  def testTrainingLoopWithInfeed(self):
    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())

    def my_net(iters):
      def body(loss, x):
        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(
              2,
              1,
              use_bias=True,
              kernel_initializer=init_ops.ones_initializer(),
              name='conv1')(x)
        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)
        with ops.control_dependencies([train]):
          return array_ops.identity(loss)

      loss = 0.0
      return ipu.loops.repeat(iters, body, (loss), infeed_queue)

    with ops.device('cpu'):
      iters = array_ops.placeholder(np.int32, shape=[])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[iters])

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      sess.run(variables.global_variables_initializer())
      initial_loss = sess.run(r, {iters: 1})
      final_loss = sess.run(r, {iters: 1000})
      self.assertTrue(initial_loss > final_loss)

  @test_util.deprecated_graph_mode_only
  def testSingleOutfeedRepeatNonTuple(self):

    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    def body(v):
      outfeed = outfeed_queue.enqueue(v)
      v = v + 1
      return (v, outfeed)

    def my_net(v):
      r = ipu.loops.repeat(20, body, (v))
      return r

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [4, 4])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[v])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    outfeed = outfeed_queue.dequeue()
    with session_lib.Session() as sess:
      result = sess.run(res, {v: np.ones([4, 4], np.float32)})

      self.assertAllClose(result[0], np.broadcast_to(21, [4, 4]))
      outfed = sess.run(outfeed)
      for i in range(20):
        self.assertAllClose(outfed[i], np.broadcast_to(i + 1, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testMultipleOutfeedsRepeatNonTuple(self):

    outfeed_queue1 = ipu.ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())
    outfeed_queue2 = ipu.ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    def body(v):
      outfeed1 = outfeed_queue1.enqueue(v)
      outfeed2 = outfeed_queue2.enqueue(v * 2)
      v = v + 1
      return (v, outfeed1, outfeed2)

    def my_net(v):
      r = ipu.loops.repeat(20, body, (v))
      return r

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [4, 4])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[v])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    outfeed1 = outfeed_queue1.dequeue()
    outfeed2 = outfeed_queue2.dequeue()
    with session_lib.Session() as sess:
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          'Only one IPUOutfeedQueue supported per graph'):
        result = sess.run(res, {v: np.ones([4, 4], np.float32)})

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedOutfeedRepeatNonTuple(self):
    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    def body(v, x):
      v = v + x
      outfeed = outfeed_queue.enqueue(v)
      return (v, outfeed)

    def my_net(v):
      r = ipu.loops.repeat(20, body, (v), infeed_queue)
      return r

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [4, 4])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[v])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res, {v: np.ones([4, 4], np.float32)})

      self.assertAllClose(result[0], np.broadcast_to(91, [4, 4]))
      outfed = sess.run(outfeed_queue.dequeue())
      self.assertEqual(outfed.shape, (20, 4, 4))
      self.assertAllClose(outfed[-1], result[0])
      self.assertAllClose(outfed[5], np.broadcast_to(16, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedOutfeedRepeatTuple(self):
    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])
    shape = [4, 4]

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    def body(v, im1, im2):
      v = v + im1 + im2
      outfeed = outfeed_queue.enqueue((v, im1, im2))
      return (v, outfeed)

    def my_net():
      v = constant_op.constant(0.0, shape=shape, dtype=np.float32)
      r = ipu.loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    outfed = outfeed_queue.dequeue()
    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(31, shape))
      outfed_result = sess.run(outfed)
      self.assertTrue(len(outfed_result) == 3)
      self.assertAllClose(outfed_result[0][0], np.broadcast_to(5, shape))
      self.assertAllClose(outfed_result[0][1], np.broadcast_to(11.5, shape))
      self.assertAllClose(outfed_result[0][2], np.broadcast_to(19.5, shape))
      self.assertAllClose(outfed_result[0][3], np.broadcast_to(24.5, shape))
      self.assertAllClose(outfed_result[0][4], np.broadcast_to(31, shape))

      self.assertAllClose(outfed_result[1][0], np.broadcast_to(0, shape))
      self.assertAllClose(outfed_result[1][1], np.broadcast_to(1, shape))
      self.assertAllClose(outfed_result[1][2], np.broadcast_to(2, shape))
      self.assertAllClose(outfed_result[1][3], np.broadcast_to(0, shape))
      self.assertAllClose(outfed_result[1][4], np.broadcast_to(1, shape))

      self.assertAllClose(outfed_result[2][0], np.broadcast_to(5, shape))
      self.assertAllClose(outfed_result[2][1], np.broadcast_to(5.5, shape))
      self.assertAllClose(outfed_result[2][2], np.broadcast_to(6, shape))
      self.assertAllClose(outfed_result[2][3], np.broadcast_to(5, shape))
      self.assertAllClose(outfed_result[2][4], np.broadcast_to(5.5, shape))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedOutfeedRepeatTupleLast(self):
    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])
    shape = [4, 4]

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
        next_feed_id(), outfeed_mode=ipu.ipu_outfeed_queue.IPUOutfeedMode.LAST)

    def body(v, im1, im2):
      v = v + im1 + im2
      outfeed = outfeed_queue.enqueue((v, im1, im2))
      return (v, outfeed)

    def my_net():
      v = constant_op.constant(0.0, shape=shape, dtype=np.float32)
      r = ipu.loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    outfed = outfeed_queue.dequeue()
    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(31, shape))
      outfed_result = sess.run(outfed)
      self.assertTrue(len(outfed_result) == 3)
      self.assertAllClose(outfed_result[0], np.broadcast_to(31, shape))
      self.assertAllClose(outfed_result[1], np.broadcast_to(1, shape))
      self.assertAllClose(outfed_result[2], np.broadcast_to(5.5, shape))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedOutfeedRepeatNamed(self):
    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])
    shape = [4, 4]

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    def body(v, im1, im2):
      v = v + im1 + im2
      outfeed = outfeed_queue.enqueue({"v": v, "image1": im1, "image2": im2})
      return (v, outfeed)

    def my_net():
      v = constant_op.constant(0.0, shape=shape, dtype=np.float32)
      r = ipu.loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    outfed = outfeed_queue.dequeue()
    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(31, shape))
      outfed_result = sess.run(outfed)
      self.assertTrue(len(outfed_result) == 3)
      self.assertAllClose(outfed_result["v"][0], np.broadcast_to(5, shape))
      self.assertAllClose(outfed_result["v"][1], np.broadcast_to(11.5, shape))
      self.assertAllClose(outfed_result["v"][2], np.broadcast_to(19.5, shape))
      self.assertAllClose(outfed_result["v"][3], np.broadcast_to(24.5, shape))
      self.assertAllClose(outfed_result["v"][4], np.broadcast_to(31, shape))

      self.assertAllClose(outfed_result["image1"][0], np.broadcast_to(
          0, shape))
      self.assertAllClose(outfed_result["image1"][1], np.broadcast_to(
          1, shape))
      self.assertAllClose(outfed_result["image1"][2], np.broadcast_to(
          2, shape))
      self.assertAllClose(outfed_result["image1"][3], np.broadcast_to(
          0, shape))
      self.assertAllClose(outfed_result["image1"][4], np.broadcast_to(
          1, shape))

      self.assertAllClose(outfed_result["image2"][0], np.broadcast_to(
          5, shape))
      self.assertAllClose(outfed_result["image2"][1],
                          np.broadcast_to(5.5, shape))
      self.assertAllClose(outfed_result["image2"][2], np.broadcast_to(
          6, shape))
      self.assertAllClose(outfed_result["image2"][3], np.broadcast_to(
          5, shape))
      self.assertAllClose(outfed_result["image2"][4],
                          np.broadcast_to(5.5, shape))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedOutfeedRepeatNamedLast(self):
    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])
    shape = [4, 4]

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
        next_feed_id(), outfeed_mode=ipu.ipu_outfeed_queue.IPUOutfeedMode.LAST)

    def body(v, im1, im2):
      v = v + im1 + im2
      outfeed = outfeed_queue.enqueue({"v": v, "image1": im1, "image2": im2})
      return (v, outfeed)

    def my_net():
      v = constant_op.constant(0.0, shape=shape, dtype=np.float32)
      r = ipu.loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    outfed = outfeed_queue.dequeue()
    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(31, shape))
      outfed_result = sess.run(outfed)
      self.assertTrue(len(outfed_result) == 3)
      self.assertAllClose(outfed_result["v"], np.broadcast_to(31, shape))
      self.assertAllClose(outfed_result["image1"], np.broadcast_to(1, shape))
      self.assertAllClose(outfed_result["image2"], np.broadcast_to(5.5, shape))

  @test_util.deprecated_graph_mode_only
  def testTrainingLoopWithInfeedAndOutfeedGetAll(self):

    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    def my_net(iters):
      def body(loss, x):
        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(
              2,
              1,
              use_bias=True,
              kernel_initializer=init_ops.ones_initializer(),
              name='conv1')(x)
        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)
        outfeed = outfeed_queue.enqueue(loss)
        with ops.control_dependencies([train]):
          return (array_ops.identity(loss), outfeed)

      loss = 0.0
      return ipu.loops.repeat(iters, body, (loss), infeed_queue)

    with ops.device('cpu'):
      iters = array_ops.placeholder(np.int32, shape=[])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[iters])

    outfeeds = outfeed_queue.dequeue()
    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      sess.run(variables.global_variables_initializer())
      initial_loss = sess.run(r, {iters: 1})
      final_loss = sess.run(r, {iters: 1000})
      outfed = sess.run(outfeeds)

      self.assertTrue(initial_loss > final_loss)
      self.assertTrue(outfed.shape[0], 1001)
      self.assertTrue(type(outfed) == np.ndarray)

  @test_util.deprecated_graph_mode_only
  def testTrainingLoopWithInfeedAndOutfeedGetLast(self):
    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
        next_feed_id(), outfeed_mode=ipu.ipu_outfeed_queue.IPUOutfeedMode.LAST)

    def my_net(iters):
      def body(loss, x):
        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(
              2,
              1,
              use_bias=True,
              kernel_initializer=init_ops.ones_initializer(),
              name='conv1')(x)
        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)
        outfeed = outfeed_queue.enqueue(loss)
        with ops.control_dependencies([train]):
          return (array_ops.identity(loss), outfeed)

      loss = 0.0
      return ipu.loops.repeat(iters, body, (loss), infeed_queue)

    with ops.device('cpu'):
      iters = array_ops.placeholder(np.int32, shape=[])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[iters])

    outfeeds = outfeed_queue.dequeue()
    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      sess.run(variables.global_variables_initializer())
      initial_loss = sess.run(r, {iters: 1})
      final_loss = sess.run(r, {iters: 1000})

      outfed = sess.run(outfeeds)

      self.assertTrue(initial_loss > final_loss)
      self.assertTrue(outfed == final_loss)

      # Check that a scalar is returned instead of a numpy array
      self.assertTrue(type(outfed) == np.float32)

  @test_util.deprecated_graph_mode_only
  def testTwoOutfeedsDifferentPrograms(self):

    outfeed_queue1 = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
        feed_name=next_feed_id())
    outfeed_queue2 = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
        feed_name=next_feed_id())

    def body1(v):
      outfeed = outfeed_queue1.enqueue(v)
      v = v + 1
      return (v, outfeed)

    def my_net1(v):
      r = ipu.loops.repeat(5, body1, (v))
      return r

    def body2(v):
      outfeed = outfeed_queue2.enqueue(v)
      v = v + 1
      return (v, outfeed)

    def my_net2(v):
      r = ipu.loops.repeat(7, body2, (v))
      return r

    with ops.device('cpu'):
      v1 = array_ops.placeholder(np.float32, [4, 4])
      v2 = array_ops.placeholder(np.float32, [5, 5])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res1 = ipu.ipu_compiler.compile(my_net1, inputs=[v1])
      res2 = ipu.ipu_compiler.compile(my_net2, inputs=[v2])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    outfeed1 = outfeed_queue1.dequeue()
    outfeed2 = outfeed_queue2.dequeue()
    with session_lib.Session() as sess:
      result1 = sess.run(res1, {v1: np.ones([4, 4], np.float32)})
      self.assertAllClose(result1[0], np.broadcast_to(6, [4, 4]))
      outfed1 = sess.run(outfeed1)
      for i in range(5):
        self.assertAllClose(outfed1[i], np.broadcast_to(i + 1, [4, 4]))

      result2 = sess.run(res2, {v2: np.full([5, 5], 4, np.float32)})
      self.assertAllClose(result2[0], np.broadcast_to(11, [5, 5]))
      outfed2 = sess.run(outfeed2)
      for i in range(7):
        self.assertAllClose(outfed2[i], np.broadcast_to(i + 4, [5, 5]))

  @test_util.deprecated_graph_mode_only
  def testTwoOutfeedsDifferentProgramsDelayedOutfeedRead(self):

    outfeed_queue1 = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
        feed_name=next_feed_id())
    outfeed_queue2 = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
        feed_name=next_feed_id())

    def body1(v):
      outfeed = outfeed_queue1.enqueue(v)
      v = v + 1
      return (v, outfeed)

    def my_net1(v):
      r = ipu.loops.repeat(5, body1, (v))
      return r

    def body2(v):
      outfeed = outfeed_queue2.enqueue(v)
      v = v + 1
      return (v, outfeed)

    def my_net2(v):
      r = ipu.loops.repeat(7, body2, (v))
      return r

    with ops.device('cpu'):
      v1 = array_ops.placeholder(np.float32, [4, 4])
      v2 = array_ops.placeholder(np.float32, [5, 5])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res1 = ipu.ipu_compiler.compile(my_net1, inputs=[v1])
      res2 = ipu.ipu_compiler.compile(my_net2, inputs=[v2])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    outfeed1 = outfeed_queue1.dequeue()
    outfeed2 = outfeed_queue2.dequeue()
    with session_lib.Session() as sess:
      result1 = sess.run(res1, {v1: np.ones([4, 4], np.float32)})
      self.assertAllClose(result1[0], np.broadcast_to(6, [4, 4]))
      result2 = sess.run(res2, {v2: np.full([5, 5], 4, np.float32)})
      self.assertAllClose(result2[0], np.broadcast_to(11, [5, 5]))

      outfed1 = sess.run(outfeed1)
      for i in range(5):
        self.assertAllClose(outfed1[i], np.broadcast_to(i + 1, [4, 4]))
      outfed2 = sess.run(outfeed2)
      for i in range(7):
        self.assertAllClose(outfed2[i], np.broadcast_to(i + 4, [5, 5]))

  @test_util.deprecated_graph_mode_only
  def testTwoOutfeedsDifferentProgramsSameFeedName(self):

    outfeed_queue1 = ipu.ipu_outfeed_queue.IPUOutfeedQueue(feed_name="a")
    outfeed_queue2 = ipu.ipu_outfeed_queue.IPUOutfeedQueue(feed_name="a")

    def body1(v):
      outfeed = outfeed_queue1.enqueue(v)
      v = v + 1
      return (v, outfeed)

    def my_net1(v):
      r = ipu.loops.repeat(5, body1, (v))
      return r

    def body2(v):
      outfeed = outfeed_queue2.enqueue(v)
      v = v + 1
      return (v, outfeed)

    def my_net2(v):
      r = ipu.loops.repeat(7, body2, (v))
      return r

    with ops.device('cpu'):
      v1 = array_ops.placeholder(np.float32, [4, 4])
      v2 = array_ops.placeholder(np.float32, [5, 5])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res1 = ipu.ipu_compiler.compile(my_net1, inputs=[v1])
      res2 = ipu.ipu_compiler.compile(my_net2, inputs=[v2])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    outfeed1 = outfeed_queue1.dequeue()
    outfeed2 = outfeed_queue2.dequeue()
    with session_lib.Session() as sess:
      result1 = sess.run(res1, {v1: np.ones([4, 4], np.float32)})
      with self.assertRaisesRegexp(errors.FailedPreconditionError,
                                   'Outfeed with id=\'a\' already exists'):
        result2 = sess.run(res2, {v2: np.full([5, 5], 4, np.float32)})

  @test_util.deprecated_graph_mode_only
  def testInfeedUsingDatasetWithNestedDictNotUnpacked(self):
    x = {
        "x0": np.ones(shape=[2], dtype=np.float32),
        "x1": np.ones(shape=[2], dtype=np.float32)
    }
    y = np.ones(shape=[2], dtype=np.float32)
    ds = dataset_ops.Dataset.from_tensor_slices((x, y))
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(
        ds, feed_name=next_feed_id())

    def body(total, x, y):
      total += x["x0"] + x["x1"] + y
      return total

    def my_net():
      r = ipu.loops.repeat(2, body, [0.0], infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)

    self.assertEqual(result, [6.0])

  @test_util.deprecated_graph_mode_only
  def testInfeedUsingDatasetWithOnlyDictIsUnpacked(self):
    x = {
        "x0": np.ones(shape=[2], dtype=np.float32),
        "x1": np.ones(shape=[2], dtype=np.float32)
    }
    ds = dataset_ops.Dataset.from_tensor_slices((x, ))
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(
        ds, feed_name=next_feed_id())

    def body(total, x0, x1):
      total += x0 + x1
      return total

    def my_net():
      r = ipu.loops.repeat(2, body, [0.0], infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)

    self.assertEqual(result, [4.0])


if __name__ == "__main__":
  googletest.main()
