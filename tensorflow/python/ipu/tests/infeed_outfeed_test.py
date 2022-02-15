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

import signal
import subprocess
import sys
from threading import Thread
import numpy as np
from absl.testing import parameterized

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent


class InfeedOutfeedTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  @parameterized.parameters([True, False])
  @test_util.deprecated_graph_mode_only
  def testSingleInfeedRepeatNonTuple(self, latency):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])
    assert isinstance(latency, bool)
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(
        dataset, optimise_latency=latency)

    def body(v, x):
      v = v + x
      return v

    def my_net(v):
      r = ipu.loops.repeat(20, body, (v), infeed_queue)
      return r

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [4, 4])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[v])

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res, {v: np.ones([4, 4], np.float32)})
      self.assertAllClose(result[0], np.broadcast_to(91, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedRepeatNonTupleFiniteDataset(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(10,
                                                  shape=[4, 4],
                                                  repeat=False)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def body(v, x):
      v = v + x
      return v

    def my_net(v):
      r = ipu.loops.repeat(10, body, (v), infeed_queue)
      return r

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [4, 4])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[v])

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res, {v: np.ones([4, 4], np.float32)})
      self.assertAllClose(result[0], np.broadcast_to(46, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedRepeatTuple(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def body(v, im1, im2):
      v = v + im1 + im2
      return v

    def my_net():
      v = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
      r = ipu.loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(31, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedRepeatTupleMerge(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.optimizations.merge_infeed_io_copies = True
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def body(v, im1, im2):
      v = v + im1 + im2
      return v

    def my_net():
      v = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
      r = ipu.loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(31, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedRepeatNamed(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return {"a": image_1, "b": image_2}

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

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

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(4, [4, 4]))
      self.assertAllClose(result[1], np.broadcast_to(27, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedMultipleRepeats(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(2, shape=[4, 4])

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def body(v, x):
      v = v + x
      return v

    def my_net():
      v = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
      r = ipu.loops.repeat(5, body, [v], infeed_queue)
      r = ipu.loops.repeat(5, body, [r], infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(5, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedWhileLoopNonTuple(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

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

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res, {v: np.ones([4, 4], np.float32)})
      self.assertAllClose(result[0], np.broadcast_to(91, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedWhileLoopTuple(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

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

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res, {v: np.ones([4, 4], np.float32)})
      self.assertAllClose(result[0], np.broadcast_to(129.5, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedMultipleRuns(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def program(iters):
      def body(v, x):
        v = v + x
        return v

      def my_net():
        v = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
        r = ipu.loops.repeat(iters, body, (v), infeed_queue)
        return r

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        return ipu.ipu_compiler.compile(my_net)

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
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset1 = tu.create_single_increasing_dataset(20, shape=[4, 4])
    dataset2 = tu.create_single_increasing_dataset(3, shape=[4, 4])

    infeed_queue1 = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset1)
    infeed_queue2 = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset2)

    def program(iters, infeed_queue):
      def body(v, x):
        v = v + x
        return v

      def my_net():
        v = constant_op.constant(0.0, shape=[4, 4], dtype=np.float32)
        r = ipu.loops.repeat(iters, body, (v), infeed_queue)
        return r

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        return ipu.ipu_compiler.compile(my_net)

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
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])
    dataset = dataset.batch(10, drop_remainder=False)
    with self.assertRaisesRegex(ValueError, r'Output shape \((\?|None),'):
      ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

  @test_util.deprecated_graph_mode_only
  def testMultipleInitializations(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    _ = infeed_queue.initializer
    with self.assertRaisesRegex(
        ValueError,
        'The IPUInfeedQueue `initializer` function can only be accessed once.'
    ):
      _ = infeed_queue.initializer

  def _testDatasetExceptionTerminates(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    # Normally we would use the deprecated_graph_mode_only decorator but this
    # function is being called inside a subprocess independently.
    with context.graph_mode():
      BAD_PATH = 'this/path/doesnt/exist/'
      dataset = readers.FixedLengthRecordDataset([BAD_PATH], 100)
      dataset = dataset.map(
          lambda f: parsing_ops.decode_raw(f, dtypes.float32)[0])
      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(infeed_queue._dequeue, [])  # pylint: disable=protected-access
      with session_lib.Session() as sess:
        sess.run(infeed_queue.initializer)
        sess.run(r)

  def testDatasetExceptionTerminates(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    TIMEOUT = 10
    cmd = [
        sys.executable, __file__, "{}.{}".format(
            InfeedOutfeedTest.__name__,
            InfeedOutfeedTest._testDatasetExceptionTerminates.__name__)
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, timeout=TIMEOUT)
    self.assertEqual(result.returncode, -signal.SIGABRT)
    error = result.stderr.decode()
    self.assertIn("An infeed dataset iterator has failed", error)

  @test_util.deprecated_graph_mode_only
  def testTrainingLoopWithInfeed(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def my_net(iters):
      def body(loss, x):
        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
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

  @parameterized.parameters([True, False])
  @test_util.deprecated_graph_mode_only
  def testMultipleOutfeedEnqueue(self, latency):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
        optimise_latency=latency)

    def body(v):
      outfeed = outfeed_queue.enqueue(v)
      outfeed = outfeed_queue.enqueue(v)
      v = v + 1
      return (v, outfeed)

    def my_net(v):
      r = ipu.loops.repeat(20, body, (v))
      return r

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [4, 4])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      with self.assertRaises(ValueError):
        ipu.ipu_compiler.compile(my_net, inputs=[v])

  @test_util.deprecated_graph_mode_only
  def testMultipleOutfeedEnqueueDifferentGraphs(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def body(v):
      outfeed = outfeed_queue.enqueue(v)
      v = v + 1
      return (v, outfeed)

    def my_net(v):
      r = ipu.loops.repeat(20, body, (v))
      return r

    with ops.Graph().as_default():
      with ops.device('cpu'):
        v = array_ops.placeholder(np.float32, [4, 4])

      self.assertFalse(outfeed_queue.enqueued)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        ipu.ipu_compiler.compile(my_net, inputs=[v])

      self.assertTrue(outfeed_queue.enqueued)

    with ops.Graph().as_default():
      with ops.device('cpu'):
        v = array_ops.placeholder(np.float32, [4, 4])

      # Not enqueued in the current graph.
      self.assertFalse(outfeed_queue.enqueued)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        ipu.ipu_compiler.compile(my_net, inputs=[v])

      self.assertTrue(outfeed_queue.enqueued)

  @test_util.deprecated_graph_mode_only
  def testSingleOutfeedRepeatNonTuple(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

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

    outfeed = outfeed_queue.dequeue()
    with session_lib.Session() as sess:
      result = sess.run(res, {v: np.ones([4, 4], np.float32)})

      self.assertAllClose(result[0], np.broadcast_to(21, [4, 4]))
      outfed = sess.run(outfeed)

      for i in range(20):
        self.assertAllClose(outfed[i], np.broadcast_to(i + 1, [4, 4]))

  @test_util.deprecated_graph_mode_only
  def testMultipleOutfeedsInSameGraph(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    outfeed_queue1 = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
    outfeed_queue2 = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def inner_body(v):
      outfeed = outfeed_queue2.enqueue(v)
      v = v + 1
      return v, outfeed

    def body(v):
      outfeed = outfeed_queue1.enqueue(v)
      v = ipu.loops.repeat(10, inner_body, v)
      return v, outfeed

    def my_net(v):
      r = ipu.loops.repeat(10, body, v)
      return r

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[v])

    dequeued1 = outfeed_queue1.dequeue()
    dequeued2 = outfeed_queue2.dequeue()

    with session_lib.Session() as sess:
      sess.run(res, {v: 0.0})
      out1, out2 = sess.run([dequeued1, dequeued2])
      self.assertAllEqual(np.arange(0, 100, step=10), out1)
      self.assertAllEqual(np.arange(0, 100, step=1), out2)

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedOutfeedRepeatNonTuple(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4])

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

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
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])
    shape = [4, 4]

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

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
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])
    shape = [4, 4]

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
        outfeed_mode=ipu.ipu_outfeed_queue.IPUOutfeedMode.LAST)

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
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])
    shape = [4, 4]

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

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

      self.assertAllClose(outfed_result["image1"][0],
                          np.broadcast_to(0, shape))
      self.assertAllClose(outfed_result["image1"][1],
                          np.broadcast_to(1, shape))
      self.assertAllClose(outfed_result["image1"][2],
                          np.broadcast_to(2, shape))
      self.assertAllClose(outfed_result["image1"][3],
                          np.broadcast_to(0, shape))
      self.assertAllClose(outfed_result["image1"][4],
                          np.broadcast_to(1, shape))

      self.assertAllClose(outfed_result["image2"][0],
                          np.broadcast_to(5, shape))
      self.assertAllClose(outfed_result["image2"][1],
                          np.broadcast_to(5.5, shape))
      self.assertAllClose(outfed_result["image2"][2],
                          np.broadcast_to(6, shape))
      self.assertAllClose(outfed_result["image2"][3],
                          np.broadcast_to(5, shape))
      self.assertAllClose(outfed_result["image2"][4],
                          np.broadcast_to(5.5, shape))

  @test_util.deprecated_graph_mode_only
  def testSingleInfeedOutfeedRepeatNamedLast(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(3, shape=[4, 4])
    shape = [4, 4]

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)

    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
        outfeed_mode=ipu.ipu_outfeed_queue.IPUOutfeedMode.LAST)

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
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def my_net(iters):
      def body(loss, x):
        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
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
      self.assertTrue(isinstance(outfed, np.ndarray))

  @test_util.deprecated_graph_mode_only
  def testTrainingLoopWithInfeedAndOutfeedGetLast(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(10, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
        outfeed_mode=ipu.ipu_outfeed_queue.IPUOutfeedMode.LAST)

    def my_net(iters):
      def body(loss, x):
        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
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
      self.assertTrue(isinstance(outfed, np.float32))

  @test_util.deprecated_graph_mode_only
  def testTwoOutfeedsDifferentPrograms(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    outfeed_queue1 = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
    outfeed_queue2 = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

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
  def testOutfeedNonTensorOutputs(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def body1():
      with variable_scope.variable_scope("", use_resource=True):
        w = variable_scope.get_variable(
            "w",
            dtype=np.float32,
            shape=[1],
            initializer=init_ops.constant_initializer(2.0))
      outfeed = outfeed_queue.enqueue({101: 1, 2020: w})
      return outfeed

    def net():
      r = ipu.loops.repeat(5, body1)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(net, inputs=[])

    outfeed = outfeed_queue.dequeue()
    with session_lib.Session() as sess:
      tu.move_variable_initialization_to_cpu()
      sess.run(variables.global_variables_initializer())

      sess.run(res)
      outfed = sess.run(outfeed)

      for i in range(5):
        self.assertAllClose(outfed[101][i], 1)
        self.assertAllClose(outfed[2020][i], [2.0])

  @test_util.deprecated_graph_mode_only
  def testTwoOutfeedsDifferentProgramsDelayedOutfeedRead(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    outfeed_queue1 = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
    outfeed_queue2 = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

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
  def testInfeedUsingDatasetWithNestedDictNotUnpacked(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    x = {
        "x0": np.ones(shape=[2], dtype=np.float32),
        "x1": np.ones(shape=[2], dtype=np.float32)
    }
    y = np.ones(shape=[2], dtype=np.float32)
    ds = dataset_ops.Dataset.from_tensor_slices((x, y))
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(ds)

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
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    x = {
        "x0": np.ones(shape=[2], dtype=np.float32),
        "x1": np.ones(shape=[2], dtype=np.float32)
    }
    ds = dataset_ops.Dataset.from_tensor_slices((x,))
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(ds)

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

  @test_util.deprecated_graph_mode_only
  def testInfeedDeleteBeforeInitializeShouldRaiseException(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(10)
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    delete_op = infeed_queue.deleter
    with session_lib.Session() as sess:
      with self.assertRaisesRegex(errors_impl.NotFoundError,
                                  "Infeed with id="):
        sess.run(delete_op)

  @test_util.deprecated_graph_mode_only
  def testInfeedRestart(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    # Note: This is not something that we encourage or need to support,
    # but it is the current behaviour that we document in this test:
    # The infeed can be restarted by calling the `deleter` and then the
    # `initializer` again.

    def data_gen():
      for i in range(5):
        yield i

    dataset = dataset_ops.Dataset.from_generator(data_gen, np.float32, ())
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    init_op = infeed_queue.initializer
    delete_op = infeed_queue.deleter

    def body(v, x):
      v = v + x
      return v

    def my_net(v):
      r = ipu.loops.repeat(5, body, (v), infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      [res] = ipu.ipu_compiler.compile(my_net, inputs=[0.0])

    with session_lib.Session() as sess:
      for _ in range(2):
        sess.run(init_op)
        self.assertEqual(sum(range(5)), sess.run(res))
        sess.run(delete_op)

  @test_util.deprecated_graph_mode_only
  def testInfeedOutfeedContinuousDequeuing(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    num_iterations = 1000
    dataset = tu.create_single_increasing_dataset(num_iterations, shape=[1])

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def body(x):
      return outfeed_queue.enqueue(x)

    def my_net():
      return ipu.loops.repeat(num_iterations, body, [], infeed_queue)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    outfed = outfeed_queue.dequeue()

    with session_lib.Session() as sess:

      def dequeue(result):
        while len(result) != 1000:
          r = sess.run(outfed)
          if r.size:
            result.extend(list(r.flatten()))

      sess.run(infeed_queue.initializer)
      r = []
      dequeue_thread = Thread(target=dequeue, args=[r])
      dequeue_thread.start()
      sess.run(res)
      dequeue_thread.join()
      self.assertAllClose(r, range(0, 1000))

  @test_util.deprecated_graph_mode_only
  def testInfeedOutfeedContinuousDequeuingGetLastBeforeEnqueued(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    num_iterations = 1000
    dataset = tu.create_single_increasing_dataset(num_iterations, shape=[1])

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
        outfeed_mode=ipu.ipu_outfeed_queue.IPUOutfeedMode.LAST)

    def body(x):
      return outfeed_queue.enqueue(x)

    def my_net():
      return ipu.loops.repeat(num_iterations, body, [], infeed_queue)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    outfed = outfeed_queue.dequeue()

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      with self.assertRaisesRegex(errors.FailedPreconditionError,
                                  r'Trying to get the last value from an'):
        sess.run(outfed)
        sess.run(res)

  @test_util.deprecated_graph_mode_only
  def testCannotFeedInt64(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = dataset_ops.Dataset.range(5)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def body(v, x):
      v = v + math_ops.cast(x, np.int32)
      return v

    def my_net():
      r = ipu.loops.repeat(5, body, (0,), infeed_queue)
      return r

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      ipu.ipu_compiler.compile(my_net, inputs=[])

    with session_lib.Session() as sess:
      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          "Unsupported datatype int64 on index 0 of feed operation"):
        sess.run(infeed_queue.initializer)

  @test_util.deprecated_graph_mode_only
  def testFeedBools(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    left = [False, False, True, True]
    right = [False, True, False, True]
    dataset = dataset_ops.Dataset.from_tensor_slices((left, right))
    dataset = dataset.batch(2, drop_remainder=True)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def body(l, r):
      return outfeed_queue.enqueue(math_ops.logical_and(l, r))

    def my_net():
      return ipu.loops.repeat(2, body, infeed_queue=infeed_queue)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    dequeued = outfeed_queue.dequeue()
    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      sess.run(res)
      out = sess.run(dequeued)
      self.assertAllEqual(np.logical_and(left, right), np.concatenate(out))

  @test_util.deprecated_graph_mode_only
  def testHashTableInDataPipeline(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], np.int32)
    table = lookup_ops.StaticHashTableV1(
        initializer=lookup_ops.KeyValueTensorInitializer(keys, values),
        default_value=-1)

    dataset = dataset_ops.Dataset.from_tensor_slices(
        ["brain brain tank salad surgery".split()])
    dataset = dataset.map(table.lookup)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def my_net():
      return infeed_queue._dequeue()  # pylint: disable=protected-access

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      [res] = ipu.ipu_compiler.compile(my_net)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      sess.run(table.initializer)
      self.assertAllEqual([0, 0, -1, 1, 2], sess.run(res))

  @test_util.deprecated_graph_mode_only
  def testFeedInt8(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(10, dtype=np.int8, shape=[])

    def m(x):
      x = x - 5
      return (x, math_ops.cast(x, np.uint8))

    dataset = dataset.map(m)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def body(x1, x2):
      x1 = math_ops.cast(x1, np.float16)
      x2 = math_ops.cast(x2, np.float32)
      x1 = x1 + 1
      x2 = x2 - 1
      x1 = math_ops.cast(x1, np.int8)
      x2 = math_ops.cast(x2, np.uint8)
      return outfeed_queue.enqueue((x1, x2))

    def my_net():
      return ipu.loops.repeat(10, body, infeed_queue=infeed_queue)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    dequeued = outfeed_queue.dequeue()
    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      sess.run(res)
      out = sess.run(dequeued)
      self.assertAllEqual([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5], out[0])
      self.assertAllEqual([250, 251, 252, 253, 254, 255, 0, 1, 2, 3], out[1])

  @test_util.deprecated_graph_mode_only
  def test8bitOps(self):
    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    dataset = tu.create_single_increasing_dataset(1, dtype=np.int8, shape=[10])

    dataset = dataset.map(lambda x: (x, math_ops.cast(x, np.uint8)))

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    def body(x1, x2):
      results = [
          array_ops.identity(x1),
          array_ops.identity(x2),
          array_ops.reshape(x1, shape=[2, 5]),
          array_ops.reshape(x2, shape=[2, 5]),
          array_ops.expand_dims(x1, axis=0),
          array_ops.expand_dims(x2, axis=0),
          array_ops.broadcast_to(x1, shape=[10, 10]),
          array_ops.broadcast_to(x2, shape=[10, 10]),
      ]

      return outfeed_queue.enqueue(results)

    def my_net():
      return ipu.loops.repeat(1, body, infeed_queue=infeed_queue)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      res = ipu.ipu_compiler.compile(my_net, inputs=[])

    dequeued = outfeed_queue.dequeue()
    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      sess.run(res)
      out = sess.run(dequeued)
      self.assertAllEqual(np.full([1, 10], 0), out[0])
      self.assertAllEqual(np.full([1, 10], 0), out[1])
      self.assertAllEqual(np.full([1, 2, 5], 0), out[2])
      self.assertAllEqual(np.full([1, 2, 5], 0), out[3])
      self.assertAllEqual(np.full([1, 1, 10], 0), out[4])
      self.assertAllEqual(np.full([1, 1, 10], 0), out[5])
      self.assertAllEqual(np.full([1, 10, 10], 0), out[6])
      self.assertAllEqual(np.full([1, 10, 10], 0), out[7])


if __name__ == "__main__":
  googletest.main()
