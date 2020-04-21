# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import ipu
from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
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

      bl = ['*convert*/Cast*']
      report.assert_compute_sets_not_in_blacklist(bl)
      report.assert_tensor_input_names("input_a", "x", "y")

  def testBatchNormalizeFp16(self):
    with self.session() as sess:
      a = array_ops.placeholder(np.float16, [4, 64, 64, 4], name="input_a")

      def my_graph(a):
        with ops.device("/device:IPU:0"):
          with variable_scope.variable_scope("", use_resource=True):

            beta = variable_scope.get_variable(
                "x",
                dtype=np.float16,
                shape=[4],
                initializer=init_ops.constant_initializer(0.0))
            gamma = variable_scope.get_variable(
                "y",
                dtype=np.float16,
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

      bl = ['*FusedBatchNorm*/Cast*']
      report.assert_compute_sets_not_in_blacklist(bl)
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

      bl = ['*convert*/Cast*']
      report.assert_compute_sets_not_in_blacklist(bl)

      report.assert_tensor_input_names("input_a", "x", "y")

  def testBatchNormalizeFusedFp16(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          a = array_ops.placeholder(np.float16, [4, 64, 64, 4], name="input_a")

          def my_graph(a):
            beta = variable_scope.get_variable(
                "x",
                dtype=np.float16,
                shape=[4],
                initializer=init_ops.constant_initializer(0.0))
            gamma = variable_scope.get_variable(
                "y",
                dtype=np.float16,
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

        bl = ['*FusedBatchNorm*/Cast*']
        report.assert_compute_sets_not_in_blacklist(bl)
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

      bl = ['*convert*/Cast*']
      report.assert_compute_sets_not_in_blacklist(bl)
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

      bl = ['*convert*/Cast*']
      report.assert_compute_sets_not_in_blacklist(bl)
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

      bl = ['*convert*/Cast*']
      report.assert_compute_sets_not_in_blacklist(bl)
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

      bl = ['*convert*/Cast*']
      report.assert_compute_sets_not_in_blacklist(bl)
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


if __name__ == "__main__":
  googletest.main()
