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

import numpy as np
import pva

from tensorflow.python import ipu
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent

from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops


class EmbeddingLookupTest(test_util.TensorFlowTestCase):
  def validate_output(self, input_tensor, indices, output_tensor):
    for i, value in enumerate(indices):
      if isinstance(value, np.int32):
        self.assertEqual(tuple(output_tensor[i]), tuple(input_tensor[value]))
      else:
        self.validate_output(input_tensor, value, output_tensor[i])

  def _validate_gradient_output(self, indices, grads, output_tensor, scale,
                                visited):
    for i, value in enumerate(indices):
      if isinstance(value, np.int32):
        visited.append(value)
        self.assertEqual(tuple(grads[i] * scale), tuple(output_tensor[value]))
      else:
        self._validate_gradient_output(value, grads[i], output_tensor, scale,
                                       visited)

  def validate_gradient_output(self, indices, grads, output_tensor, scale):
    visited = []
    # Check all the indices contain the corresponding gradient slice.
    self._validate_gradient_output(indices, grads, output_tensor, scale,
                                   visited)
    # Check the other values are 0:
    for i, output_slice in enumerate(output_tensor):
      if i not in visited:
        self.assertFalse(output_slice.any())
      else:
        self.assertTrue(output_slice.any())

  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testGather(self):
    def my_net(w, i):
      out = ipu.ops.embedding_ops.embedding_lookup(w, i)
      self.assertEqual(out.shape, (8, 200))
      return [out]

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [8])
      w = array_ops.placeholder(np.float32, [12000, 200])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[w, i])

    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 1472
    cfg.configure_ipu_system()
    with sl.Session() as sess:
      i_h = np.arange(0, 8)
      w_h = np.arange(2400000).reshape([12000, 200])

      result = sess.run(r, {i: i_h, w: w_h})
      self.assertAllClose(result[0], np.take(w_h, i_h, axis=0))
      self.assertEqual(result[0].shape, (8, 200))

  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testAutoFlatten(self):
    with self.session() as sess:
      with ops.device('cpu'):
        x1 = array_ops.placeholder(np.int32, shape=[3, 4, 2])

      def network(x1):
        with variable_scope.variable_scope("vs", use_resource=True):
          x = variable_scope.get_variable(
              "x",
              shape=[100, 16],
              dtype=np.float32,
              initializer=init_ops.random_normal_initializer(stddev=0.1))
        out = ipu.ops.embedding_ops.embedding_lookup(x, x1)
        self.assertEqual(out.shape, (3, 4, 2, 16))
        return out, x, x1

      with ops.device("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(network, inputs=[x1])

      cfg = ipu.config.IPUConfig()
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()
      sess.run(variables.global_variables_initializer())
      out, input_tensor, indices = sess.run(
          r, {
              x1: [[[10, 11], [12, 13], [14, 15], [16, 17]],
                   [[20, 21], [22, 23], [24, 25], [26, 27]],
                   [[30, 31], [32, 33], [34, 35], [36, 37]]]
          })
      self.assertEqual(out.shape, (3, 4, 2, 16))
      self.validate_output(input_tensor, indices, out)

  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testWithResourceVariable(self):
    with self.session() as sess:
      with ops.device('cpu'):
        x1 = array_ops.placeholder(np.int32, shape=[10])
        lr = array_ops.placeholder(np.float32, shape=[])

      def network(x1, lr):
        with variable_scope.variable_scope("vs", use_resource=True):
          x = variable_scope.get_variable(
              "x",
              shape=[100, 16],
              dtype=np.float32,
              initializer=init_ops.random_normal_initializer(stddev=0.1))
        g1 = ipu.ops.embedding_ops.embedding_lookup(x, x1)
        self.assertEqual(g1.shape, (10, 16))
        optimizer = gradient_descent.GradientDescentOptimizer(lr)
        train = optimizer.minimize(g1)
        return g1, x, x1, train

      with ops.device("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(network, inputs=[x1, lr])

      cfg = ipu.config.IPUConfig()
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()
      sess.run(variables.global_variables_initializer())
      out, input_tensor, indices = sess.run(
          r, {
              x1: [4, 8, 15, 16, 23, 42, 8, 4, 15, 16],
              lr: 0.1,
          })
      self.assertEqual(out.shape, (10, 16))
      self.validate_output(input_tensor, indices, out)

  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testWithResourceVariableAutoFlatten(self):
    with self.session() as sess:

      def network(x1, lr):
        with variable_scope.variable_scope("vs", use_resource=True):
          x = variable_scope.get_variable(
              "x",
              shape=[100, 16],
              dtype=np.float32,
              initializer=init_ops.random_normal_initializer(stddev=0.1))
        g1 = ipu.ops.embedding_ops.embedding_lookup(x, x1)
        self.assertEqual(g1.shape, (3, 4, 2, 16))
        optimizer = gradient_descent.GradientDescentOptimizer(lr)
        train = optimizer.minimize(g1)
        return g1, x, x1, train

      with ops.device('cpu'):
        x1 = array_ops.placeholder(np.int32, shape=[3, 4, 2])
        lr = array_ops.placeholder(np.float32, shape=[])

      with ops.device("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(network, inputs=[x1, lr])

      cfg = ipu.config.IPUConfig()
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()
      sess.run(variables.global_variables_initializer())
      out, input_tensor, indices = sess.run(
          r, {
              x1: [[[10, 11], [12, 13], [14, 15], [16, 17]],
                   [[20, 21], [22, 23], [24, 25], [26, 27]],
                   [[30, 31], [32, 33], [34, 35], [36, 37]]],
              lr:
              0.1,
          })
      self.assertEqual(out.shape, (3, 4, 2, 16))
      self.validate_output(input_tensor, indices, out)

  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testGradient(self):
    with self.session() as sess:
      with ops.device('cpu'):
        x1 = array_ops.placeholder(np.int32, shape=[3, 4, 2])
        grads = array_ops.placeholder(np.float32, shape=[3, 4, 2, 16])
        lr = array_ops.placeholder(np.float32, shape=[])

      def network(x1, grads, lr):
        with variable_scope.variable_scope("vs", use_resource=True):
          x = variable_scope.get_variable(
              "x",
              shape=[100, 16],
              dtype=np.float32,
              initializer=init_ops.random_normal_initializer(stddev=0.1))
        out = gen_popops_ops.ipu_multi_update_add(array_ops.zeros_like(x),
                                                  updates=grads,
                                                  indices=x1,
                                                  scale=lr)
        self.assertEqual(out.shape, x.shape)
        return out, x1, grads

      with ops.device("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(network, inputs=[x1, grads, lr])

      cfg = ipu.config.IPUConfig()
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()
      sess.run(variables.global_variables_initializer())
      out, indices, gradient = sess.run(
          r, {
              x1: [[[10, 11], [12, 13], [14, 15], [16, 17]],
                   [[20, 21], [22, 23], [24, 25], [26, 27]],
                   [[30, 31], [32, 33], [34, 35], [36, 37]]],
              grads:
              np.random.rand(*grads.shape),
              lr:
              0.1,
          })
      self.validate_gradient_output(indices, gradient, out, 0.1)

  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def test4D(self):
    def my_net(w, i):
      out = ipu.ops.embedding_ops.embedding_lookup(w, i)
      self.assertEqual(out.shape, (8, 2, 200, 4))
      return [out]

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [8, 2])
      w = array_ops.placeholder(np.float32, [32, 200, 4])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[w, i])

    cfg = ipu.config.IPUConfig()
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()
    with sl.Session() as sess:
      i_h = np.arange(0, 16).reshape([8, 2])
      w_h = np.arange(25600).reshape([32, 200, 4])

      result = sess.run(r, {i: i_h, w: w_h})
      self.assertAllClose(result[0], np.take(w_h, i_h, axis=0))
      self.assertEqual(result[0].shape, (8, 2, 200, 4))

  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testEmbeddingUpdateWithMatMul(self):
    def my_net(i):
      w = variable_scope.get_variable(
          "w",
          shape=[1000, 200],
          dtype=np.float32,
          initializer=init_ops.random_normal_initializer(stddev=0.1))
      out = ipu.ops.embedding_ops.embedding_lookup(w, i)
      out = math_ops.matmul(w, out, transpose_b=True)

      optimizer = gradient_descent.GradientDescentOptimizer(1e-3)
      training_op = optimizer.minimize(out)
      return training_op

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [8])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[i])

    cfg = ipu.config.IPUConfig()
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()
    with sl.Session() as sess:
      i_h = np.arange(0, 8)

      sess.run(variables.global_variables_initializer())
      # Just checking that graph construction succeeds and the program runs
      sess.run(r, {i: i_h})

  @tu.test_may_use_ipus_or_model(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testSerializedEmbeddingLookup(self):
    with sl.Session() as sess:

      def body(table, indices):
        return ipu.ops.embedding_ops.embedding_lookup(table,
                                                      indices,
                                                      serialization_factor=4)

      with ops.device('cpu'):
        table = array_ops.placeholder(np.float16, [2000, 4, 4, 8])
        indices = array_ops.placeholder(np.int32, [4, 4, 8])

      cfg = ipu.config.IPUConfig()
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(body, inputs=[table, indices])

      table_h = np.arange(128, dtype=np.float16).reshape([4, 4, 8]) * np.ones(
          [2000, 4, 4, 8], dtype=np.float16)
      indices_h = np.random.random_integers(0, 2000, [4, 4, 8])
      result = sess.run(res, {table: table_h, indices: indices_h})
      self.assertAllClose(result[0], np.take(table_h, indices_h, axis=0))

  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testSerializedEmbeddingLookupDoesntDivide(self):
    with sl.Session():

      def body(table, indices):
        return ipu.ops.embedding_ops.embedding_lookup(table,
                                                      indices,
                                                      serialization_factor=7)

      with ops.device('cpu'):
        table = array_ops.placeholder(np.float16, [2000, 4, 4, 8])
        indices = array_ops.placeholder(np.int32, [4, 4, 8])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        with self.assertRaisesRegex(
            ValueError,
            r"The serialization_factor \(7\) must divide the size of the 0th "
            r"dimension of params \(2000\)."):
          ipu.ipu_compiler.compile(body, inputs=[table, indices])

  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testReallocationInGradientWhenInputAllocatedForMatmul(self):
    # Tests the behaviour when the input is allocated for a different op.
    # The input should be cloned into a correctly mapped tensor, to avoid
    # memory spikes.
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.tiles_per_ipu = 1472
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device('cpu'):
        x = array_ops.placeholder(np.float32, shape=[16, 16])
        updates = array_ops.placeholder(np.float32, shape=[1024, 16])
        indices = array_ops.placeholder(np.int32, shape=[1024])

      def model(x, updates, indices):
        with variable_scope.variable_scope("vs", use_resource=True):
          lhs = variable_scope.get_variable(
              "lhs",
              shape=[1024, 16],
              dtype=np.float32,
              initializer=init_ops.random_normal_initializer(stddev=0.1))
        x = math_ops.matmul(lhs, x)
        return gen_popops_ops.ipu_multi_update_add(x,
                                                   updates=updates,
                                                   indices=indices,
                                                   scale=1)

      with ops.device("/device:IPU:0"):
        result = ipu.ipu_compiler.compile(model, inputs=[x, updates, indices])

      fd = {
          x: np.random.rand(*x.shape),
          updates: np.random.rand(*updates.shape),
          indices: np.random.randint(1024, size=indices.shape),
      }

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()
      sess.run(result, feed_dict=fd)

      # Large memory spikes are generated when the input for a MultiUpdateAdd
      # is not mapped in the scheme expected by poplibs.
      report = pva.openReport(report_helper.find_report())
      self.assert_max_tile_memory(report, 2096, tolerance=0.1)

  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testOutputWhenInputAllocatedForAnotherOp(self):
    # Tests the output values for the code path where the input is allocated
    # for a different op. In this path the input is cloned into a correctly
    # mapped tensor.
    with self.session() as sess:
      with ops.device('cpu'):
        x = array_ops.placeholder(np.float32, shape=[4, 4])
        indices = array_ops.placeholder(np.int32, shape=[8])

      def model(x, indices):
        with variable_scope.variable_scope("vs", use_resource=True):
          lhs = variable_scope.get_variable(
              "lhs",
              shape=[8, 4],
              dtype=np.float32,
              initializer=init_ops.ones_initializer())
        x = math_ops.matmul(lhs, x)
        gen_popops_ops.ipu_multi_slice(x, indices)
        return x

      with ops.device("/device:IPU:0"):
        result, = ipu.ipu_compiler.compile(model, inputs=[x, indices])

      fd = {
          x: [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]],
          indices: list(range(8)),
      }

      sess.run(variables.global_variables_initializer())

      out = sess.run(result, feed_dict=fd)
      self.assertAllClose(out, [[12, 16, 20, 24]] * 8)

  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testGradientOutputWhenInputAllocatedForAnotherOp(self):
    # Tests the output values for the code path where the input is allocated
    # for a different op. In this path the input is cloned into a correctly
    # mapped tensor, and output is copied back to the original input tensor.
    with self.session() as sess:
      with ops.device('cpu'):
        x = array_ops.placeholder(np.float32, shape=[4, 4])
        updates = array_ops.placeholder(np.float32, shape=[8, 4])
        indices = array_ops.placeholder(np.int32, shape=[8])
        scale = 5

      def model(x, updates, indices):
        with variable_scope.variable_scope("vs", use_resource=True):
          lhs = variable_scope.get_variable(
              "lhs",
              shape=[8, 4],
              dtype=np.float32,
              initializer=init_ops.ones_initializer())
        x = math_ops.matmul(lhs, x)
        return gen_popops_ops.ipu_multi_update_add(x,
                                                   updates=updates,
                                                   indices=indices,
                                                   scale=scale)

      with ops.device("/device:IPU:0"):
        result, = ipu.ipu_compiler.compile(model, inputs=[x, updates, indices])

      fd = {
          x: np.zeros(x.shape),
          indices: list(reversed(range(8))),
          updates: np.random.rand(*updates.shape),
      }

      sess.run(variables.global_variables_initializer())

      out = sess.run(result, feed_dict=fd)
      expected_out = np.flip(fd[updates], axis=0) * scale
      self.assertAllClose(out, expected_out)


if __name__ == "__main__":
  googletest.main()
