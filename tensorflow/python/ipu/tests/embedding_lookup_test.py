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

import numpy as np

from tensorflow.python import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
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

  def _validate_gradient_output(self, indices, grads, output_tensor, visited):
    for i, value in enumerate(indices):
      if isinstance(value, np.int32):
        visited.append(value)
        self.assertEqual(tuple(grads[i]), tuple(output_tensor[value]))
      else:
        self._validate_gradient_output(value, grads[i], output_tensor, visited)

  def validate_gradient_output(self, indices, grads, output_tensor):
    visited = []
    # Check all the indices contain the corresponding gradient slice.
    self._validate_gradient_output(indices, grads, output_tensor, visited)
    # Check the other values are 0:
    for i, output_slice in enumerate(output_tensor):
      if i not in visited:
        self.assertFalse(output_slice.any())
      else:
        self.assertTrue(output_slice.any())

  @test_util.deprecated_graph_mode_only
  def testGather(self):
    def my_net(w, i):
      out = ipu.ops.embedding_ops.embedding_lookup(w,
                                                   i,
                                                   min_encoding_size=1200)
      self.assertEqual(out.shape, (8, 200))
      return [out]

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [8])
      w = array_ops.placeholder(np.float32, [12000, 200])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[w, i])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)
    with sl.Session() as sess:
      i_h = np.arange(0, 8)
      w_h = np.arange(2400000).reshape([12000, 200])

      result = sess.run(r, {i: i_h, w: w_h})
      self.assertAllClose(result[0], np.take(w_h, i_h, axis=0))
      self.assertEqual(result[0].shape, (8, 200))

  def testAutoFlatten(self):
    with self.session() as sess:
      with ops.device('cpu'):
        x1 = array_ops.placeholder(np.int32, shape=[3, 4, 2])

      def network(x, x1):
        out = ipu.ops.embedding_ops.embedding_lookup(x, x1)
        self.assertEqual(out.shape, (3, 4, 2, 16))
        return out, x, x1

      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):
          x = variable_scope.get_variable("x",
                                          initializer=random_ops.random_normal(
                                              [100, 16], stddev=0.1))
          r = ipu.ipu_compiler.compile(network, inputs=[x, x1])

      sess.run(variables.variables_initializer([x]))
      out, input_tensor, indices = sess.run(
          r, {
              x1: [[[10, 11], [12, 13], [14, 15], [16, 17]],
                   [[20, 21], [22, 23], [24, 25], [26, 27]],
                   [[30, 31], [32, 33], [34, 35], [36, 37]]]
          })
      self.assertEqual(out.shape, (3, 4, 2, 16))
      self.validate_output(input_tensor, indices, out)

  def testWithResourceVariable(self):
    with self.session() as sess:
      with ops.device('cpu'):
        x1 = array_ops.placeholder(np.int32, shape=[10])
        lr = array_ops.placeholder(np.int32, shape=[])

      def network(x, x1, lr):
        g1 = ipu.ops.embedding_ops.embedding_lookup(x, x1)
        self.assertEqual(g1.shape, (10, 16))
        optimizer = gradient_descent.GradientDescentOptimizer(lr)
        train = optimizer.minimize(g1)
        return g1, x, x1, train

      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):
          x = variable_scope.get_variable("x",
                                          initializer=random_ops.random_normal(
                                              [100, 16], stddev=0.1))
          r = ipu.ipu_compiler.compile(network, inputs=[x, x1, lr])

      sess.run(variables.variables_initializer([x]))
      out, input_tensor, indices = sess.run(
          r, {
              x1: [4, 8, 15, 16, 23, 42, 8, 4, 15, 16],
              lr: 0.1,
          })
      self.assertEqual(out.shape, (10, 16))
      self.validate_output(input_tensor, indices, out)

  def testWithResourceVariableAutoFlatten(self):
    with self.session() as sess:
      with ops.device('cpu'):
        x1 = array_ops.placeholder(np.int32, shape=[3, 4, 2])
        lr = array_ops.placeholder(np.int32, shape=[])

      def network(x, x1, lr):
        g1 = ipu.ops.embedding_ops.embedding_lookup(x, x1)
        self.assertEqual(g1.shape, (3, 4, 2, 16))
        optimizer = gradient_descent.GradientDescentOptimizer(lr)
        train = optimizer.minimize(g1)
        return g1, x, x1, train

      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):
          x = variable_scope.get_variable("x",
                                          initializer=random_ops.random_normal(
                                              [100, 16], stddev=0.1))
          r = ipu.ipu_compiler.compile(network, inputs=[x, x1, lr])

      sess.run(variables.variables_initializer([x]))
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

  def testGradient(self):
    with self.session() as sess:
      with ops.device('cpu'):
        x1 = array_ops.placeholder(np.int32, shape=[3, 4, 2])
        grads = array_ops.placeholder(np.float32, shape=[3, 4, 2, 16])

      def network(x, x1, grads):
        out = gen_popops_ops.ipu_multi_update(array_ops.zeros_like(x),
                                              gradient=grads,
                                              indices=x1)
        self.assertEqual(out.shape, x.shape)
        return out, x1, grads

      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):
          x = variable_scope.get_variable("x",
                                          initializer=random_ops.random_normal(
                                              [100, 16], stddev=0.1))
          r = ipu.ipu_compiler.compile(network, inputs=[x, x1, grads])

      sess.run(variables.variables_initializer([x]))
      out, indices, gradient = sess.run(
          r, {
              x1: [[[10, 11], [12, 13], [14, 15], [16, 17]],
                   [[20, 21], [22, 23], [24, 25], [26, 27]],
                   [[30, 31], [32, 33], [34, 35], [36, 37]]],
              grads:
              np.random.rand(*grads.shape)
          })
      self.assertEqual(out.shape, x.shape)
      self.validate_gradient_output(indices, gradient, out)


if __name__ == "__main__":
  googletest.main()
