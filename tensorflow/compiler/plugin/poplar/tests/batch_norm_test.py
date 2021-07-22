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

import pva
import numpy as np
import test_utils as tu

from tensorflow.python import ipu
from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ipu.config import IPUConfig
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
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.configure_ipu_system()

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

      report_json = ReportJSON(self, sess)
      out = ipu.ipu_compiler.compile(my_graph, [a])
      sess.run(variables.global_variables_initializer())

      report_json.reset()
      report_helper.clear_reports()

      result = sess.run(out, {a: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([1, 4, 64, 64, 4]))

      report_json.parse_log()
      report_json.assert_tensor_input_names("input_a", "x", "y")

    report = pva.openReport(report_helper.find_report())
    bl = ['*convert*/Cast*']
    self.assert_compute_sets_not_in_blacklist(report, bl)

  def testBatchNormalizeFp16(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.configure_ipu_system()

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

      report_json = ReportJSON(self, sess)
      out = ipu.ipu_compiler.compile(my_graph, [a])
      sess.run(variables.global_variables_initializer())

      report_json.reset()
      report_helper.clear_reports()

      result = sess.run(out, {a: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([1, 4, 64, 64, 4]))

      report_json.parse_log()
      report_json.assert_tensor_input_names("input_a", "x", "y")

    report = pva.openReport(report_helper.find_report())
    bl = ['*FusedBatchNorm*/Cast*']
    self.assert_compute_sets_not_in_blacklist(report, bl)

  def testBatchNormalizeFused(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.configure_ipu_system()

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

      report_json = ReportJSON(self, sess)
      out = ipu.ipu_compiler.compile(my_graph, [a])
      sess.run(variables.global_variables_initializer())

      report_json.reset()
      report_helper.clear_reports()

      result, _, _ = sess.run(out, {a: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))
      report_json.parse_log()
      report_json.assert_tensor_input_names("input_a", "x", "y")

    report = pva.openReport(report_helper.find_report())
    bl = ['*convert*/Cast*']
    self.assert_compute_sets_not_in_blacklist(report, bl)

  def testBatchNormalizeFusedFp16(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.configure_ipu_system()

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

        report_json = ReportJSON(self, sess)
        out = ipu.ipu_compiler.compile(my_graph, [a])
        sess.run(variables.global_variables_initializer())

        report_json.reset()
        report_helper.clear_reports()

        result, _, _ = sess.run(out, {a: np.zeros([4, 64, 64, 4])})
        self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

        report_json.parse_log()
        report_json.assert_tensor_input_names("input_a", "x", "y")

    report = pva.openReport(report_helper.find_report())
    bl = ['*FusedBatchNorm*/Cast*']
    self.assert_compute_sets_not_in_blacklist(report, bl)

  def testBatchNormalizeLayer(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          a = array_ops.placeholder(np.float32, [4, 64, 64, 4], name="input_a")

          normed = layers_norm.batch_normalization(a, fused=False)

      report_json = ReportJSON(self, sess)
      sess.run(variables.global_variables_initializer())

      report_json.reset()
      report_helper.clear_reports()

      result = sess.run(normed, {a: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

      report_json.parse_log()
      report_json.assert_tensor_input_names("input_a")

    report = pva.openReport(report_helper.find_report())
    bl = ['*convert*/Cast*']
    self.assert_compute_sets_not_in_blacklist(report, bl)

  def testBatchNormalizeLayerWithStableStatistics(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.norms.use_stable_statistics = True
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          a = array_ops.placeholder(np.float32, [4, 64, 64, 4], name="input_a")
          normed = layers_norm.batch_normalization(a, training=True)

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
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          a = array_ops.placeholder(np.float32, [4, 64, 64, 4], name="input_a")

          normed = layers_norm.batch_normalization(a, fused=True)

      report_json = ReportJSON(self, sess)
      sess.run(variables.global_variables_initializer())

      report_json.reset()
      report_helper.clear_reports()

      result = sess.run(normed, {a: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

      report_json.parse_log()
      report_json.assert_tensor_input_names("input_a")

    report = pva.openReport(report_helper.find_report())
    bl = ['*convert*/Cast*']
    self.assert_compute_sets_not_in_blacklist(report, bl)

  def testBatchNormalizeLayerFp16(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          a = array_ops.placeholder(np.float16, [4, 64, 64, 4], name="input_a")

          normed = layers_norm.batch_normalization(a, fused=False)

      report_json = ReportJSON(self, sess)
      sess.run(variables.global_variables_initializer())

      report_json.reset()
      report_helper.clear_reports()

      result = sess.run(normed, {a: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

      report_json.parse_log()
      report_json.assert_tensor_input_names("input_a")

    report = pva.openReport(report_helper.find_report())
    bl = ['*convert*/Cast*']
    self.assert_compute_sets_not_in_blacklist(report, bl)

  def testBatchNormalizeLayerFusedFp16(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          a = array_ops.placeholder(np.float16, [4, 64, 64, 4], name="input_a")

          normed = layers_norm.batch_normalization(a, fused=True)

      report_json = ReportJSON(self, sess)
      sess.run(variables.global_variables_initializer())

      report_json.reset()
      report_helper.clear_reports()

      result = sess.run(normed, {a: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result, np.zeros([4, 64, 64, 4]))

      report_json.parse_log()
      report_json.assert_tensor_input_names("input_a")

    report = pva.openReport(report_helper.find_report())
    bl = ['*convert*/Cast*']
    self.assert_compute_sets_not_in_blacklist(report, bl)

  def testBatchNormalizeLayerFusedTrainingFp16(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.configure_ipu_system()

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

      sess.run(variables.global_variables_initializer())
      result = sess.run([normed, train], {a: np.zeros([4, 64, 64, 4])})
      self.assertAllClose(result[0], np.zeros([4, 64, 64, 4]))


if __name__ == "__main__":
  googletest.main()
