# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import os
import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.compiler.tests import xla_test
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.compiler.xla import xla
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import constant_op


class WideConstExpansionTest(xla_test.XLATestCase):
  def testCheckMaxTileSize(self):
    with self.session() as sess:
      dtype = np.float32
      shape = (1024, 2048)
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          a = variable_scope.get_variable(
              "a",
              shape=shape,
              initializer=init_ops.constant_initializer(2),
              dtype=dtype)
        pb = array_ops.placeholder(shape=shape, dtype=dtype, name="b")
        c = constant_op.constant(4, shape=shape, dtype=dtype, name="c")
        output = a + pb + c

      report = tu.ReportJSON(self, sess)
      report.reset()

      sess.run(variables.global_variables_initializer())

      report.parse_log()
      report.assert_max_tile_memory(6240)

      out = sess.run(output, {pb: np.ones(shape=shape, dtype=dtype)})
      self.assertAllClose(np.full(shape, 7, dtype=dtype), out)

      report.parse_log()
      report.assert_max_tile_memory(23462)

  def testWideConstantWithAllocationTarget(self):
    with self.session() as sess:
      # This test will fail if the dynamic slice is not mapped correctly.
      dtype = np.float32
      shape = (512, 2, 2048)

      def my_net(y):
        def cond(i, x, y):
          del x
          del y
          return i < 2

        def body(i, x, y):
          s = array_ops.slice(x, [i, i, i], [1, 1, 2048])
          y = y + math_ops.reduce_mean(s)
          x = x + constant_op.constant(1, shape=shape, dtype=dtype)
          i = i + 1
          return (i, x, y)

        i = 0
        c = constant_op.constant(4, shape=shape, dtype=dtype, name="c")
        return control_flow_ops.while_loop(cond, body, (i, c, y), name='')[2]

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      report = tu.ReportJSON(self, sess)
      report.reset()

      y = sess.run(r, {y: [10]})
      self.assertAllClose(y[0], [19])

      report.parse_log(assert_len=4)

      ok = [
          '__seed*', 'Copy_*_to_*', 'Slice/dynamic-slice*/dynamicSlice',
          'Mean/reduce', 'Mean/multiply', 'add*/add*/Add',
          'add_*/fusion/Op/Add'
      ]
      report.assert_all_compute_sets_and_list(ok)

      report.assert_max_tile_memory(8936)
      report.assert_always_live_memory(299760)

  def testCheckMaxTileSizePadding(self):
    with self.session() as sess:

      def my_graph(a, b):
        with variable_scope.variable_scope("vs", use_resource=True):
          weights = variable_scope.get_variable(
              "x",
              dtype=np.float16,
              shape=[1024, 1024],
              initializer=init_ops.constant_initializer(0.0))

        a = array_ops.pad(a, [[0, 0], [23, 1]])
        a = a - b * 0.1
        mm1 = math_ops.matmul(a, weights, name="mm1")
        return mm1

      pa = array_ops.placeholder(np.float16, [1024, 1000], name="a")
      pb = array_ops.placeholder(np.float16, [1024, 1024], name="a")

      with ops.device("/device:IPU:0"):
        out = ipu_compiler.compile(my_graph, [pa, pb])

      report = tu.ReportJSON(self, sess)
      report.reset()

      tu.move_variable_initialization_to_cpu()
      sess.run(variables.global_variables_initializer())
      report.reset()

      out = sess.run(out, {pa: np.ones(pa.shape), pb: np.ones(pb.shape)})
      self.assertAllClose(np.zeros(pb.shape), out[0])

      report.parse_log()
      report.assert_max_tile_memory(77762)

  def testCheckMaxTileSizePadding2(self):
    with self.session() as sess:

      def my_graph(a, b):
        with variable_scope.variable_scope("vs", use_resource=True):
          weights = variable_scope.get_variable(
              "x",
              dtype=np.float16,
              shape=[64, 64],
              initializer=init_ops.constant_initializer(1.0))
        a = math_ops.matmul(a, weights, name="mm1")
        a = array_ops.pad(a, [[0, 0], [4935, 1]], constant_values=64)
        return a + b

      pa = array_ops.placeholder(np.float16, [64, 64], name="a")
      pb = array_ops.placeholder(np.float16, [64, 5000], name="a")

      with ops.device("/device:IPU:0"):
        out = ipu_compiler.compile(my_graph, [pa, pb])

      report = tu.ReportJSON(self, sess)
      report.reset()

      tu.move_variable_initialization_to_cpu()
      sess.run(variables.global_variables_initializer())
      report.reset()

      out = sess.run(out, {pa: np.ones(pa.shape), pb: np.ones(pb.shape)})
      self.assertAllClose(np.full(pb.shape, 65.0), out[0])

      report.parse_log()
      report.assert_max_tile_memory(2594)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
