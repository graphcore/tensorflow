#  Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import test_util as tu

from tensorflow.contrib import ipu
from tensorflow.keras import layers
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent

from tensorflow.contrib.ipu import ipu_compiler
from tensorflow.contrib.ipu import ipu_infeed_queue
from tensorflow.contrib.ipu import ipu_outfeed_queue
from tensorflow.contrib.ipu import loops


def next_feed_id():
  result = 'feed' + str(next_feed_id.feed_count)
  next_feed_id.feed_count += 1
  return result


next_feed_id.feed_count = 0


class InfeedPrefetchTest(test_util.TensorFlowTestCase):
  def testSingleInfeedRepeatNonTuple(self):
    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, next_feed_id(), data_to_prefetch=4)

    def body(v, x):
      v = v + x
      return (v)

    def my_net(v):
      r = loops.repeat(20, body, (v), infeed_queue)
      return r

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [4, 4])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = ipu_compiler.compile(my_net, inputs=[v])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res, {v: np.ones([4, 4], np.float32)})
      self.assertAllClose(result[0], np.broadcast_to(91, [4, 4]))

  def testSingleInfeedRepeatNonTupleFiniteDataset(self):
    dataset = tu.create_single_increasing_dataset(
        10, shape=[4, 4], repeat=False)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, next_feed_id(), data_to_prefetch=2)

    def body(v, x):
      v = v + x
      return (v)

    def my_net(v):
      r = loops.repeat(10, body, (v), infeed_queue)
      return r

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [4, 4])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = ipu_compiler.compile(my_net, inputs=[v])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res, {v: np.ones([4, 4], np.float32)})
      self.assertAllClose(result[0], np.broadcast_to(46, [4, 4]))

  def testSingleInfeedRepeatTuple(self):
    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, next_feed_id(), data_to_prefetch=3)

    def body(v, im1, im2):
      v = v + im1 + im2
      return (v)

    def my_net():
      v = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
      r = loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = ipu_compiler.compile(my_net, inputs=[])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(31, [4, 4]))

  def testSingleInfeedRepeatTupleMerge(self):
    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, next_feed_id(), data_to_prefetch=2)

    def body(v, im1, im2):
      v = v + im1 + im2
      return (v)

    def my_net():
      v = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
      r = loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = ipu_compiler.compile(my_net, inputs=[])

    cfg = ipu.utils.create_ipu_config(merge_infeed_io_copies=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(31, [4, 4]))

  def testSingleInfeedRepeatNamed(self):
    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return {"a": image_1, "b": image_2}

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, next_feed_id(), data_to_prefetch=2)

    #Note how the parameters are swapped around.
    def body(v1, v2, b, a):
      v1 = v1 + a
      v2 = v2 + b
      return (v1, v2)

    def my_net():
      v1 = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
      v2 = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
      r = loops.repeat(5, body, [v1, v2], infeed_queue)
      return r

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = ipu_compiler.compile(my_net, inputs=[])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(4, [4, 4]))
      self.assertAllClose(result[1], np.broadcast_to(27, [4, 4]))

  def testSingleInfeedMultipleRepeats(self):
    dataset = tu.create_single_increasing_dataset(2, shape=[4, 4])

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, next_feed_id(), data_to_prefetch=2)

    def body(v, x):
      v = v + x
      return (v)

    def my_net():
      v = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)

      # Iters was 5 in the original test but we must fully consume the infeed otherwise the state will be incorrect.
      r = loops.repeat(6, body, [v], infeed_queue)
      r = loops.repeat(6, body, [r], infeed_queue)
      return r

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = ipu_compiler.compile(my_net, inputs=[])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(6, [4, 4]))

  def testSingleInfeedWhileLoopNonTuple(self):
    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, next_feed_id(), data_to_prefetch=2)

    def cond(i, v):
      return i < 20

    def body(i, v, x):
      v = v + x
      return (i + 1, v)

    def my_net(v):
      i = 0
      r = loops.while_loop(cond, body, (i, v), infeed_queue)
      return r[1]

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [4, 4])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = ipu_compiler.compile(my_net, inputs=[v])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res, {v: np.ones([4, 4], np.float32)})
      self.assertAllClose(result[0], np.broadcast_to(91, [4, 4]))

  def testSingleInfeedWhileLoopTuple(self):
    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, next_feed_id(), data_to_prefetch=2)

    def cond(i, v):
      return i < 20

    def body(i, v, im1, im2):
      v = v + im1 + im2
      return (i + 1, v)

    def my_net(v):
      i = 0
      r = loops.while_loop(cond, body, (i, v), infeed_queue)
      return r[1]

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [4, 4])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = ipu_compiler.compile(my_net, inputs=[v])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res, {v: np.ones([4, 4], np.float32)})
      self.assertAllClose(result[0], np.broadcast_to(129.5, [4, 4]))

  def testSingleInfeedMultipleRuns(self):
    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())

    def program(iters):
      def body(v, x):
        v = v + x
        return (v)

      def my_net():
        v = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
        r = loops.repeat(iters, body, (v), infeed_queue)
        return r

      with ipu.ops.ipu_scope("/device:IPU:0"):
        return ipu_compiler.compile(my_net)

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(program(0))
      self.assertAllClose(result[0], np.broadcast_to(0, [4, 4]))
      #The iterator has not moved - next element should be all 1s.
      result = sess.run(program(2))
      self.assertAllClose(result[0], np.broadcast_to(1, [4, 4]))
      # The iterator has moved - in the next two iterations it should pull 2 and 3.
      result = sess.run(program(2))
      self.assertAllClose(result[0], np.broadcast_to(5, [4, 4]))
      #The iterator has moved - in the next two iterations it should pull 4 and 5.
      result = sess.run(program(2))
      self.assertAllClose(result[0], np.broadcast_to(9, [4, 4]))

  def testTwoInfeedsDifferentPrograms(self):
    dataset1 = tu.create_single_increasing_dataset(20, shape=[4, 4])
    dataset2 = tu.create_single_increasing_dataset(3, shape=[4, 4])

    infeed_queue1 = ipu_infeed_queue.IPUInfeedQueue(
        dataset1, feed_name=next_feed_id(), data_to_prefetch=2)
    infeed_queue2 = ipu_infeed_queue.IPUInfeedQueue(
        dataset2, feed_name=next_feed_id(), data_to_prefetch=2)

    def program(iters, infeed_queue):
      def body(v, x):
        v = v + x
        return (v)

      def my_net():
        v = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
        r = loops.repeat(iters, body, (v), infeed_queue)
        return r

      with ipu.ops.ipu_scope("/device:IPU:0"):
        return ipu_compiler.compile(my_net)

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

      # This was in the original test but I think it should be a user error in this case.
      # The point is that we should now be starting at the same point that we left off in
      # the previous session but since we have prefetched but only partially consumed the
      # prefetched data we are not at the expected point as we assume the prefetched data
      # is fully consumed each time.

      result = sess.run(program(5, infeed_queue1))
      self.assertAllClose(result[0], np.broadcast_to(40, [4, 4]))
      result = sess.run(program(5, infeed_queue2))
      self.assertAllClose(result[0], np.broadcast_to(4, [4, 4]))

  def testTrainingLoopWithInfeed(self):
    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, next_feed_id(), data_to_prefetch=5)

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
      return loops.repeat(iters, body, (loss), infeed_queue)

    with ops.device('cpu'):
      iters = array_ops.placeholder(np.int32, shape=[])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[iters])

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      sess.run(variables.global_variables_initializer())
      initial_loss = sess.run(r, {iters: 1})
      final_loss = sess.run(r, {iters: 1000})
      self.assertTrue(initial_loss > final_loss)


if __name__ == "__main__":
  googletest.main()
