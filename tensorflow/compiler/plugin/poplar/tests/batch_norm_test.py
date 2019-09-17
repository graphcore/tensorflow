# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

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
      x = array_ops.placeholder(np.float32, [4, 64, 64, 4], name="a")

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

          b_mean, b_var = nn.moments(x, [0, 1, 2], name='moments')

          normed = nn.batch_normalization(x, b_mean, b_var, beta, gamma, 1e-3)

      report = ReportJSON(self, sess)
      report.reset()

      sess.run(variables.global_variables_initializer())
      result = sess.run(normed, {x: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

      report.parse_log()

      bl = ['*convert*/Cast*']
      report.assert_compute_sets_not_in_blacklist(bl)

  def testBatchNormalizeFp16(self):
    with self.session() as sess:
      x = array_ops.placeholder(np.float16, [4, 64, 64, 4], name="a")

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

          b_mean, b_var = nn.moments(x, [0, 1, 2], name='moments')

          normed = nn.batch_normalization(x, b_mean, b_var, beta, gamma, 1e-3)

      report = ReportJSON(self, sess)
      report.reset()

      sess.run(variables.global_variables_initializer())
      result = sess.run(normed, {x: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

      report.parse_log()

      bl = ['*convert*/Cast*']
      report.assert_compute_sets_not_in_blacklist(bl)

  def testBatchNormalizeFused(self):
    with self.session() as sess:
      x = array_ops.placeholder(np.float32, [4, 64, 64, 4], name="a")

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

          b_mean, b_var = nn.moments(x, [0, 1, 2], name='moments')

          normed = nn.fused_batch_norm(x,
                                       gamma,
                                       beta,
                                       b_mean,
                                       b_var,
                                       is_training=False)

      report = ReportJSON(self, sess)
      report.reset()

      sess.run(variables.global_variables_initializer())
      result, _, _ = sess.run(normed, {x: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

      report.parse_log()

      bl = ['*convert*/Cast*']
      report.assert_compute_sets_not_in_blacklist(bl)

  def testBatchNormalizeFusedFp16(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          x = array_ops.placeholder(np.float16, [4, 64, 64, 4], name="a")

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

          b_mean, b_var = nn.moments(x, [0, 1, 2], name='moments')

          normed = nn.fused_batch_norm(x,
                                       gamma,
                                       beta,
                                       b_mean,
                                       b_var,
                                       is_training=False)

      report = ReportJSON(self, sess)
      report.reset()

      sess.run(variables.global_variables_initializer())
      result, _, _ = sess.run(normed, {x: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

      report.parse_log()

      bl = ['*convert*/Cast*']
      report.assert_compute_sets_not_in_blacklist(bl)

  def testBatchNormalizeLayer(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          x = array_ops.placeholder(np.float32, [4, 64, 64, 4], name="a")

          normed = layers_norm.batch_normalization(x, fused=False)

      report = ReportJSON(self, sess)
      report.reset()

      sess.run(variables.global_variables_initializer())
      result = sess.run(normed, {x: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

      report.parse_log()

      bl = ['*convert*/Cast*']
      report.assert_compute_sets_not_in_blacklist(bl)

  def testBatchNormalizeFusedLayer(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          x = array_ops.placeholder(np.float32, [4, 64, 64, 4], name="a")

          normed = layers_norm.batch_normalization(x, fused=True)

      report = ReportJSON(self, sess)
      report.reset()

      sess.run(variables.global_variables_initializer())
      result = sess.run(normed, {x: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

      report.parse_log()

      bl = ['*convert*/Cast*']
      report.assert_compute_sets_not_in_blacklist(bl)

  def testBatchNormalizeLayerFp16(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          x = array_ops.placeholder(np.float16, [4, 64, 64, 4], name="a")

          normed = layers_norm.batch_normalization(x, fused=False)

      report = ReportJSON(self, sess)
      report.reset()

      sess.run(variables.global_variables_initializer())
      result = sess.run(normed, {x: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

      report.parse_log()

      bl = ['*convert*/Cast*']
      report.assert_compute_sets_not_in_blacklist(bl)

  def testBatchNormalizeLayerFusedFp16(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          x = array_ops.placeholder(np.float16, [4, 64, 64, 4], name="a")

          normed = layers_norm.batch_normalization(x, fused=True)

      report = ReportJSON(self, sess)
      report.reset()

      sess.run(variables.global_variables_initializer())
      result = sess.run(normed, {x: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

      report.parse_log()

      bl = ['*convert*/Cast*']
      report.assert_compute_sets_not_in_blacklist(bl)

  def testBatchNormalizeLayerFusedTrainingFp16(self):
    with self.session() as sess:
      # This test checks for the correct behaviour in batch norm grad when
      # perofrming training, but the batch norm attribute `training` is False
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          x = array_ops.placeholder(np.float16, [4, 64, 64, 4], name="a")
          normed = layers_norm.batch_normalization(x,
                                                   fused=True,
                                                   training=False)
        loss = math_ops.reduce_sum(normed)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)

      ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())
      result = sess.run([normed, train], {x: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result[0], np.zeros([4, 64, 64, 4]))


if __name__ == "__main__":
  googletest.main()
