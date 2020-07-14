# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python import cast as tf_cast
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent
from tensorflow.python.layers import normalization as layers_norm

from test_utils import ReportJSON


class IpuXlaBatchNormTest(xla_test.XLATestCase):
  def testBatchNormalize(self):
    with self.session() as sess:
      a = array_ops.placeholder(np.float32, [4, 64, 64, 4], name="input_a")

      def my_graph(a):
        with ops.device("/device:IPU:0"):
          with variable_scope.variable_scope("", use_resource=True):

            beta = variable_scope.get_variable(
                "x",
                dtype=np.float32,
                shape=[4],
                initializer=init_ops.constant_initializer(0.0))
            gamma = variable_scope.get_variable(
                "y",
                dtype=np.float32,
                shape=[4],
                initializer=init_ops.constant_initializer(1.0))

            b_mean, b_var = nn.moments(a, [0, 1, 2], name='moments')

            normed = nn.batch_normalization(a, b_mean, b_var, beta, gamma,
                                            1e-3)
            return normed

      report = ReportJSON(self, sess)
      out = ipu.ipu_compiler.compile(my_graph, [a])
      sess.run(variables.global_variables_initializer())

      report.reset()
      result = sess.run(out, {a: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([1, 4, 64, 64, 4]))

      report.parse_log()
      report.assert_tensor_input_names("input_a", "x", "y")

  def testBatchNormalizeFp16(self):
    with self.session() as sess:
      a = array_ops.placeholder(np.float16, [4, 64, 64, 4], name="input_a")

      def my_graph(a):
        with ops.device("/device:IPU:0"):
          with variable_scope.variable_scope("", use_resource=True):

            beta = variable_scope.get_variable(
                "x",
                dtype=np.float32,
                shape=[4],
                initializer=init_ops.constant_initializer(0.0))
            gamma = variable_scope.get_variable(
                "y",
                dtype=np.float32,
                shape=[4],
                initializer=init_ops.constant_initializer(1.0))

            b_mean, b_var = nn.moments(a, [0, 1, 2], name='moments')
            b_mean_32 = tf_cast(b_mean, np.float32)
            b_var_32 = tf_cast(b_var, np.float32)

            normed = nn.batch_normalization(a, b_mean_32, b_var_32, beta,
                                            gamma, 1e-3)
            return normed

      report = ReportJSON(self, sess)
      out = ipu.ipu_compiler.compile(my_graph, [a])
      sess.run(variables.global_variables_initializer())
      report.reset()
      result = sess.run(out, {a: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([1, 4, 64, 64, 4]))

      report.parse_log()
      report.assert_tensor_input_names("input_a", "x", "y")

  def testBatchNormalizeFused(self):
    with self.session() as sess:
      a = array_ops.placeholder(np.float32, [4, 64, 64, 4], name="input_a")

      def my_graph(a):
        with ops.device("/device:IPU:0"):
          with variable_scope.variable_scope("", use_resource=True):

            beta = variable_scope.get_variable(
                "x",
                dtype=np.float32,
                shape=[4],
                initializer=init_ops.constant_initializer(0.0))
            gamma = variable_scope.get_variable(
                "y",
                dtype=np.float32,
                shape=[4],
                initializer=init_ops.constant_initializer(1.0))

            b_mean, b_var = nn.moments(a, [0, 1, 2], name='moments')

            normed = nn.fused_batch_norm(a,
                                         gamma,
                                         beta,
                                         b_mean,
                                         b_var,
                                         is_training=False)
            return normed

      report = ReportJSON(self, sess)
      out = ipu.ipu_compiler.compile(my_graph, [a])
      sess.run(variables.global_variables_initializer())

      report.reset()
      result, _, _ = sess.run(out, {a: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

      report.parse_log()
      report.assert_tensor_input_names("input_a", "x", "y")

  def testBatchNormalizeFusedFp16(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          a = array_ops.placeholder(np.float16, [4, 64, 64, 4], name="input_a")

          def my_graph(a):
            beta = variable_scope.get_variable(
                "x",
                dtype=np.float32,
                shape=[4],
                initializer=init_ops.constant_initializer(0.0))
            gamma = variable_scope.get_variable(
                "y",
                dtype=np.float32,
                shape=[4],
                initializer=init_ops.constant_initializer(1.0))

            b_mean, b_var = nn.moments(a, [0, 1, 2], name='moments')
            b_mean_32 = tf_cast(b_mean, np.float32)
            b_var_32 = tf_cast(b_var, np.float32)

            normed = nn.fused_batch_norm(a,
                                         gamma,
                                         beta,
                                         b_mean_32,
                                         b_var_32,
                                         is_training=False)
            return normed

        report = ReportJSON(self, sess)
        out = ipu.ipu_compiler.compile(my_graph, [a])
        sess.run(variables.global_variables_initializer())

        report.reset()
        result, _, _ = sess.run(out, {a: np.zeros([4, 64, 64, 4])})
        self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

        report.parse_log()
        report.assert_tensor_input_names("input_a", "x", "y")

  def testBatchNormalizeLayer(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          a = array_ops.placeholder(np.float32, [4, 64, 64, 4], name="input_a")

          normed = layers_norm.batch_normalization(a, fused=False)

      report = ReportJSON(self, sess)
      sess.run(variables.global_variables_initializer())

      report.reset()
      result = sess.run(normed, {a: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

      report.parse_log()
      report.assert_tensor_input_names("input_a")

  def testBatchNormalizeLayerWithStableStatistics(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          a = array_ops.placeholder(np.float32, [4, 64, 64, 4], name="input_a")
          normed = layers_norm.batch_normalization(a, training=True)

      ReportJSON(self, sess, use_stable_norm_statistics=True)
      sess.run(variables.global_variables_initializer())

      # Use a tensor with large mean to test the stability. This blows up with
      # the non-stable implementation (NaN output). Use a power-of-two that can
      # be represented exactly in float32 to make sure we work with an exact
      # mean internally.
      input_mean = 2.0**64
      inputs = input_mean * np.ones([4, 64, 64, 4])

      # y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
      # Both (x - mean) and beta_initializer are zero, so this should be zero.
      result = sess.run(normed, {a: inputs})
      self.assertAllEqual(result, np.zeros([4, 64, 64, 4]))

  def testBatchNormalizeFusedLayer(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          a = array_ops.placeholder(np.float32, [4, 64, 64, 4], name="input_a")

          normed = layers_norm.batch_normalization(a, fused=True)

      report = ReportJSON(self, sess)
      sess.run(variables.global_variables_initializer())

      report.reset()
      result = sess.run(normed, {a: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

      report.parse_log()
      report.assert_tensor_input_names("input_a")

  def testBatchNormalizeLayerFp16(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          a = array_ops.placeholder(np.float16, [4, 64, 64, 4], name="input_a")

          normed = layers_norm.batch_normalization(a, fused=False)

      report = ReportJSON(self, sess)
      sess.run(variables.global_variables_initializer())

      report.reset()
      result = sess.run(normed, {a: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

      report.parse_log()
      report.assert_tensor_input_names("input_a")

  def testBatchNormalizeLayerFusedFp16(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          a = array_ops.placeholder(np.float16, [4, 64, 64, 4], name="input_a")

          normed = layers_norm.batch_normalization(a, fused=True)

      report = ReportJSON(self, sess)
      sess.run(variables.global_variables_initializer())

      report.reset()
      result = sess.run(normed, {a: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

      report.parse_log()
      report.assert_tensor_input_names("input_a")

  def testBatchNormalizeLayerFusedTrainingFp16(self):
    with self.session() as sess:
      # This test checks for the correct behaviour in batch norm grad when
      # perofrming training, but the batch norm attribute `training` is False
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          a = array_ops.placeholder(np.float16, [4, 64, 64, 4], name="input_a")
          normed = layers_norm.batch_normalization(a,
                                                   fused=True,
                                                   training=False)
        loss = math_ops.reduce_sum(normed)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)

      ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())
      result = sess.run([normed, train], {a: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result[0], np.zeros([4, 64, 64, 4]))

  def testBatchNormalizeFusedTypes(self):
    def test_with_types(data_dtype, stats_dtype, sess):
      x_np = np.random.rand(4, 64, 64, 4).astype(data_dtype)
      scale_np = np.ones((4), dtype=stats_dtype)
      offset_np = np.zeros((4), dtype=stats_dtype)
      mean_np = np.ones((4), dtype=stats_dtype)
      inv_sd_np = np.ones((4), dtype=stats_dtype)

      x = array_ops.placeholder(data_dtype, x_np.shape, name="x")
      scale = array_ops.placeholder(stats_dtype, scale_np.shape, name="scale")
      offset = array_ops.placeholder(stats_dtype,
                                     offset_np.shape,
                                     name="offset")
      mean = array_ops.placeholder(stats_dtype, mean_np.shape, name="mean")
      inv_sd = array_ops.placeholder(stats_dtype,
                                     inv_sd_np.shape,
                                     name="variance")

      sess.run(variables.global_variables_initializer())

      # Compute forward pass.
      fwd = gen_nn_ops.fused_batch_norm_v2(x, scale, offset, mean, inv_sd)
      y, mu, sigma, _, _ = sess.run(
          fwd, {
              x: x_np,
              scale: scale_np,
              offset: offset_np,
              mean: mean_np,
              inv_sd: inv_sd_np
          })

      self.assertEqual(y.dtype, data_dtype)
      self.assertEqual(mu.dtype, stats_dtype)
      self.assertEqual(sigma.dtype, stats_dtype)

      # Compute backward pass.
      bwd = gen_nn_ops.fused_batch_norm_grad_v2(y, x, scale, mu, sigma)
      d_x, d_scale, d_offset, _, _ = sess.run(bwd, {x: x_np, scale: scale_np})

      self.assertEqual(d_x.dtype, data_dtype)
      self.assertEqual(d_scale.dtype, stats_dtype)
      self.assertEqual(d_offset.dtype, stats_dtype)

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        test_with_types(np.float32, np.float32, sess)
        test_with_types(np.float16, np.float32, sess)

      with self.assertRaisesRegex(
          TypeError, "DataType float16 not in list of allowed values"):
        test_with_types(np.float16, np.float16, sess)


if __name__ == "__main__":
  googletest.main()
