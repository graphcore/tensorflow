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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.compiler.plugin.poplar.tests.test_utils import ReportJSON
from tensorflow.python import ipu
from tensorflow.python.framework import ops
from tensorflow.python.ipu.optimizers import sharded_optimizer
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.framework import errors


class MultiIpuTest(xla_test.XLATestCase):
  def testMultiIpu(self):
    with self.session() as sess:

      def my_graph(pa, pb, pc):
        with ops.device("/device:IPU:0"):
          with ipu.scopes.ipu_shard(0):
            o1 = pa + pb

          with ipu.scopes.ipu_shard(1):
            o2 = pa + pc
            out = o1 + o2

        return [out]

      with ops.device('cpu'):
        pa = array_ops.placeholder(np.float32, [2], name="a")
        pb = array_ops.placeholder(np.float32, [2], name="b")
        pc = array_ops.placeholder(np.float32, [2], name="c")

      report = ReportJSON(self, sess, device_count_override=2)
      out = ipu.ipu_compiler.compile(my_graph, [pa, pb, pc])

      report.reset()

      fd = {pa: [1., 1.], pb: [0., 1.], pc: [1., 5.]}
      result = sess.run(out, fd)
      self.assertAllClose(result[0], [3., 8.])

      report.parse_log()
      tm = report.get_tensor_map()
      mods = tm.computation_names()
      self.assertEqual(len(mods), 1)

      tiles = tm.tile_ids(mods[0])

      self.assertEqual(len(tiles), 2)
      self.assertEqual(tiles, set((0, 1472)))

      ok = [
          '__seed*',
          'add*/add*/Add',
          'switchControlBroadcast2/*OnTileCopy',
          'Copy_*/ipu-inter-copy*/OnTileCopy',
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testMultipleConfigureIpuShouldFail(self):
    with self.session() as sess:

      def my_graph(pa, pb, pc):
        with ops.device("/device:IPU:0"):
          o1 = pa + pb
          o2 = pa + pc
          out = o1 + o2

        return [out]

      with ops.device('cpu'):
        pa = array_ops.placeholder(np.float32, [2], name="a")
        pb = array_ops.placeholder(np.float32, [2], name="b")
        pc = array_ops.placeholder(np.float32, [2], name="c")

      ipu.ipu_compiler.compile(my_graph, [pa, pb, pc])

      ReportJSON(self, sess, device_count_override=2)

      # Make sure changing the configuration of an already initialized IPU raises an exception.
      with self.assertRaises(Exception):
        configure_ipu_system(compile_ipu_code=True, io_trace=True)

  def testNotEnoughIpus(self):
    with self.session() as sess:

      def my_graph(pa, pb, pc):
        with ipu.scopes.ipu_shard(0):
          o1 = pa + pb
        with ipu.scopes.ipu_shard(1):
          o2 = pa + pc
        with ipu.scopes.ipu_shard(2):
          out = o1 + o2
          return out

      with ops.device('cpu'):
        pa = array_ops.placeholder(np.float32, [2], name="a")
        pb = array_ops.placeholder(np.float32, [2], name="b")
        pc = array_ops.placeholder(np.float32, [2], name="c")

      with ops.device("/device:IPU:0"):
        out = ipu.ipu_compiler.compile(my_graph, [pa, pb, pc])

      ReportJSON(self, sess, device_count_override=2)

      with self.assertRaisesRegex(errors.InternalError,
                                  'Trying to compile a graph for'):
        sess.run(out, {pa: [1., 1.], pb: [0., 1.], pc: [1., 5.]})

  def testMultiIpuVariables(self):
    with self.session() as sess:

      def my_graph(pa, pb, pc):
        with variable_scope.variable_scope("", use_resource=True):
          with ipu.scopes.ipu_scope("/device:IPU:0"):
            with ipu.scopes.ipu_shard(0):
              init1 = init_ops.constant_initializer([1.0, 3.0])
              v1 = variable_scope.get_variable("v1",
                                               dtype=np.float32,
                                               shape=[2],
                                               initializer=init1)
              o1 = pa + pb + v1

            with ipu.scopes.ipu_shard(1):
              init2 = init_ops.constant_initializer([1.0, 2.0])
              v2 = variable_scope.get_variable("v2",
                                               dtype=np.float32,
                                               shape=[2],
                                               initializer=init2)
              o2 = pa + pc + v2
              out = o1 + o2

        return [out]

      with ops.device('cpu'):
        pa = array_ops.placeholder(np.float32, [2], name="a")
        pb = array_ops.placeholder(np.float32, [2], name="b")
        pc = array_ops.placeholder(np.float32, [2], name="c")

      out = ipu.ipu_compiler.compile(my_graph, [pa, pb, pc])

      report = ReportJSON(self, sess, device_count_override=2)
      tu.move_variable_initialization_to_cpu()

      sess.run(variables.global_variables_initializer())
      report.reset()

      fd = {pa: [1., 1.], pb: [0., 1.], pc: [1., 5.]}
      result = sess.run(out, fd)
      self.assertAllClose(result[0], [5., 13.])

      report.parse_log()
      tm = report.get_tensor_map()
      comps = tm.computation_names()
      self.assertEqual(len(comps), 3)

      for c in comps:
        tiles = tm.tile_ids(c)
        if len(tiles) == 3:
          self.assertEqual(tiles, set((0, 1, 1472)))
        else:
          self.assertEqual(len(tiles), 0)

  def testMultiIpuTraining(self):
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

          opt = sharded_optimizer.ShardedOptimizer(
              gradient_descent.GradientDescentOptimizer(0.000001))
          train = opt.minimize(loss)

        return [loss, train]

      with ops.device('cpu'):
        inp = array_ops.placeholder(np.float32, [1, 32, 32, 4], name="data")
        lab = array_ops.placeholder(np.float32, [1, 8], name="labels")

      out = ipu.ipu_compiler.compile(my_graph, [inp, lab])

      report = ReportJSON(self, sess, device_count_override=2)
      tu.move_variable_initialization_to_cpu()

      sess.run(variables.global_variables_initializer())
      report.reset()

      fd = {inp: np.ones([1, 32, 32, 4]), lab: np.ones([1, 8])}
      sess.run(out, fd)

      events_types = report.parse_log()
      self.assertEqual(events_types[IpuTraceEvent.COMPILE_END], 1)

      cs_list = report.get_compute_sets()
      self.assertTrue(
          any(fnmatch.fnmatch(n, '*ipu-inter-copy*') for n in cs_list))

  def testConvAndBiasAddDifferentIPUs(self):
    with self.session() as sess:

      def my_graph(inp, bias):
        with ops.device("/device:IPU:0"):
          with ipu.scopes.ipu_shard(0):
            x = layers.Conv2D(8,
                              3,
                              padding='same',
                              name="conv",
                              use_bias=False)(inp)

          with ipu.scopes.ipu_shard(1):
            x = nn_ops.bias_add(x, bias, name='biasAdd')

        return x

      with ops.device('cpu'):
        inp = array_ops.placeholder(np.float32, [1, 32, 32, 4], name="data")
        bias = array_ops.placeholder(np.float32, [8], name="bias")

      out = ipu.ipu_compiler.compile(my_graph, [inp, bias])

      report = ReportJSON(self, sess, device_count_override=2)
      tu.move_variable_initialization_to_cpu()

      sess.run(variables.global_variables_initializer())
      report.reset()

      fd = {inp: np.ones([1, 32, 32, 4]), bias: np.ones([8])}
      sess.run(out, fd)

      events_types = report.parse_log()
      self.assertEqual(events_types[IpuTraceEvent.COMPILE_END], 1)

      # There is 1 piece of global exchange (apart from progId)
      wl = [
          'switchControlBroadcast*/GlobalPre',
          '*_to_/ipu-inter-copy*/GlobalPre',
          '__seed/set/setMasterSeed',
      ]
      report.assert_all_global_exchanges_and_list(wl)


if __name__ == "__main__":
  googletest.main()
