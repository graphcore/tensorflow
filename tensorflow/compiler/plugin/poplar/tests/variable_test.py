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

import os
import numpy as np
import pva
import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent


class IpuXlaVariableTest(xla_test.XLATestCase):
  def testInitializeSimpleVariables(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      with self.session() as sess:

        x = resource_variable_ops.ResourceVariable(random_ops.random_normal(
            [5, 5], stddev=0.1),
                                                   name="x")
        y = resource_variable_ops.ResourceVariable(random_ops.random_normal(
            [1], stddev=0.1),
                                                   name="y")

        sess.run(variables.global_variables_initializer())

        r1, r2 = sess.run([x, y])

        self.assertAllClose(r1, np.zeros([5, 5]), atol=1.0)
        self.assertAllClose(r2, [0.0], atol=1.0)

  def testInitializeSharedVariables(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      with self.session() as sess:
        with variable_scope.variable_scope("vs", use_resource=True):
          x = variable_scope.get_variable(
              "x",
              shape=[],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(1))

          y = variable_scope.get_variable(
              "y",
              shape=[],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(2))

        sess.run(variables.global_variables_initializer())

        r1, r2 = sess.run([x, y])

        self.assertAllClose(r1, 1)
        self.assertAllClose(r2, 2)

  def testRead(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      with self.session() as sess:
        with variable_scope.variable_scope("vs", use_resource=True):
          z = variable_scope.get_variable(
              "z",
              shape=[],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(3))

        sess.run(variables.global_variables_initializer())

        r = sess.run(z.read_value())

        self.assertAllClose(r, 3)

  def testAssign(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      with self.session() as sess:
        with variable_scope.variable_scope("vs", use_resource=True):
          z = variable_scope.get_variable(
              "z",
              shape=[],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(0))

        sess.run(variables.global_variables_initializer())

        sess.run(state_ops.assign(z, 2))
        r = sess.run(z)
        self.assertAllClose(r, 2)

        sess.run(state_ops.assign_add(z, 6))
        r = sess.run(z)
        self.assertAllClose(r, 8)

  def testGradientDescent(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):

          w = variable_scope.get_variable(
              "w",
              shape=[4, 2],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(
                  np.array([[1, 2], [3, 4], [5, 6], [7, 8]],
                           dtype=np.float32)))
          b = variable_scope.get_variable(
              "b",
              shape=[2],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(
                  np.array([2, 3], dtype=np.float32)))

        x = array_ops.placeholder(np.float32, shape=[1, 4])
        y = math_ops.matmul(x, w) + b

        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)

      sess.run(variables.global_variables_initializer())
      sess.run(train, {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})

      vw, vb = sess.run([w, b])

      self.assertAllClose(np.array(
          [[0.3, 1.3], [2.7, 3.7], [4.5, 5.5], [6.1, 7.1]], dtype=np.float32),
                          vw,
                          rtol=1e-4)

      self.assertAllClose(np.array([1.9, 2.9], dtype=np.float32),
                          vb,
                          rtol=1e-4)

  def testRepeatedGradientDescent(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):

          w = variable_scope.get_variable(
              "w",
              shape=[4, 2],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(
                  np.array([[1, 2], [3, 4], [5, 6], [7, 8]],
                           dtype=np.float32)))
          b = variable_scope.get_variable(
              "b",
              shape=[2],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(
                  np.array([2, 3], dtype=np.float32)))

        x = array_ops.placeholder(np.float32, shape=[1, 4])
        y = math_ops.matmul(x, w) + b

        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)

      sess.run(variables.global_variables_initializer())
      sess.run(train, {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      sess.run(train, {x: np.array([[1, 2, 3, 4]], dtype=np.float32)})
      sess.run(train, {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      sess.run(train, {x: np.array([[1, 2, 3, 4]], dtype=np.float32)})
      sess.run(train, {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})

      vw, vb = sess.run([w, b])

      self.assertAllClose(
          np.array([[-1.3, -0.3], [1.7, 2.7], [2.9, 3.9], [3.5, 4.5]],
                   dtype=np.float32),
          vw,
          rtol=1e-4)

      self.assertAllClose(np.array([1.5, 2.5], dtype=np.float32),
                          vb,
                          rtol=1e-4)

  def testMultipleUpdate(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):
          z = variable_scope.get_variable(
              "z",
              shape=[],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(0))

        updater = state_ops.assign_add(z, 1.0)

        sess.run(variables.global_variables_initializer())

        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)

        r = sess.run(z)
        self.assertAllClose(r, 10.0)

  def testRandomNormalInitalizer(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):
          i = init_ops.random_normal_initializer(mean=2.0, stddev=0.01)
          z = variable_scope.get_variable("z1",
                                          shape=[],
                                          dtype=np.float32,
                                          initializer=i)

      sess.run(variables.global_variables_initializer())
      report = pva.openReport(report_helper.find_report())

      o = sess.run(z)
      self.assertAllClose(o, 2.0, 0.2, 0.2)

      ok = [
          '__seed*',
          'vs/z1/Initializer/random_normal/RandomStandardNormal/fusion/normal'
      ]
      self.assert_all_compute_sets_and_list(report, ok)

  def testRandomNormalNonScalarInitalizer(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):
          i = init_ops.random_normal_initializer(mean=2.0, stddev=0.01)
          z = variable_scope.get_variable("z1",
                                          shape=[2],
                                          dtype=np.float32,
                                          initializer=i)
      sess.run(variables.global_variables_initializer())
      report = pva.openReport(report_helper.find_report())

      o = sess.run(z)
      self.assertAllClose(o, [2.0, 2.0], 0.2, 0.2)

      ok = [
          '__seed*',
          'vs/z1/Initializer/random_normal/RandomStandardNormal/fusion/normal'
      ]
      self.assert_all_compute_sets_and_list(report, ok)

  def testDefaultRandomNormalInitalizer(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):
          i = init_ops.random_normal_initializer()
          z = variable_scope.get_variable("z1",
                                          shape=[],
                                          dtype=np.float32,
                                          initializer=i)

        sess.run(variables.global_variables_initializer())
        o = sess.run(z)
        self.assertAllClose(o, 0.0, 1.0, 3.0)

  def testTruncatedNormalScalarInitalizer(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          i = init_ops.truncated_normal_initializer(mean=1.0, stddev=0.01)
          z = variable_scope.get_variable("z1",
                                          shape=[],
                                          dtype=np.float32,
                                          initializer=i)

      sess.run(variables.global_variables_initializer())
      o = sess.run(z)
      self.assertAllClose(o, 1.0, 0.2, 0.2)

    # Find of the names of compute sets
    report = pva.openReport(report_helper.find_report())

    # pylint: disable=line-too-long
    ok = [
        '__seed*',
        'z1/Initializer/truncated_normal/TruncatedNormal/truncated-normal*/truncatedNormal',
        'z1/Initializer/truncated_normal/mul/multiply.*/Op/Multiply',
        'z1/Initializer/truncated_normal/add*/Add'
    ]
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)

  def testTruncatedNormalInitalizer(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          i = init_ops.truncated_normal_initializer(mean=1.0, stddev=0.01)
          z = variable_scope.get_variable("z1",
                                          shape=[2, 4],
                                          dtype=np.float32,
                                          initializer=i)

      sess.run(variables.global_variables_initializer())
      o = sess.run(z)
      self.assertAllClose(o, np.ones((2, 4)), 0.2, 0.2)

    # Find of the names of compute sets
    report = pva.openReport(report_helper.find_report())

    # pylint: disable=line-too-long
    ok = [
        '__seed*',
        'z1/Initializer/truncated_normal/TruncatedNormal/truncated-normal*/truncatedNormal',
        'z1/Initializer/truncated_normal/scaled-inplace',
    ]
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)

  def testDefaultTruncatedNormalScalarInitalizer(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          i = init_ops.truncated_normal_initializer()
          z = variable_scope.get_variable("z1",
                                          shape=[],
                                          dtype=np.float32,
                                          initializer=i)

      sess.run(variables.global_variables_initializer())
      o = sess.run(z)
      self.assertAllClose(o, 1.0, 2.0, 2.0)

      # Find of the names of compute sets
    report = pva.openReport(report_helper.find_report())

    # pylint: disable=line-too-long
    ok = [
        '__seed*',
        'z1/Initializer/truncated_normal/TruncatedNormal/truncated-normal*/truncatedNormal'
    ]
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)

  def testDefaultTruncatedNormalInitalizer(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          i = init_ops.truncated_normal_initializer()
          z = variable_scope.get_variable("z1",
                                          shape=[2, 4],
                                          dtype=np.float32,
                                          initializer=i)

      sess.run(variables.global_variables_initializer())
      o = sess.run(z)
      self.assertAllClose(o, np.ones((2, 4)), 2.0, 2.0)

    # Find of the names of compute sets
    report = pva.openReport(report_helper.find_report())

    # pylint: disable=line-too-long
    ok = [
        '__seed*',
        'z1/Initializer/truncated_normal/TruncatedNormal/truncated-normal*/truncatedNormal'
    ]
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)

  def testUniformRandomInitalizer(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):
          i = init_ops.random_uniform_initializer(minval=-2.0, maxval=2.0)
          z = variable_scope.get_variable("z1",
                                          shape=[],
                                          dtype=np.float32,
                                          initializer=i)

      sess.run(variables.global_variables_initializer())
      report = pva.openReport(report_helper.find_report())

      o = sess.run(z)
      self.assertAllClose(o, 0.0, 2.0, 2.0)

      ok = [
          '__seed*',
          'vs/z1/Initializer/random_uniform/RandomUniform/fusion/uniform'
      ]
      self.assert_all_compute_sets_and_list(report, ok)

  def testUniformRandomNonScalarInitalizer(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):
          i = init_ops.random_uniform_initializer(minval=-2.0, maxval=2.0)
          z = variable_scope.get_variable("z1",
                                          shape=[2],
                                          dtype=np.float32,
                                          initializer=i)

      sess.run(variables.global_variables_initializer())
      report = pva.openReport(report_helper.find_report())

      o = sess.run(z)
      self.assertAllClose(o, [0.0, 0.0], 2.0, 2.0)

      ok = [
          '__seed*',
          'vs/z1/Initializer/random_uniform/RandomUniform/fusion/uniform'
      ]
      self.assert_all_compute_sets_and_list(report, ok)

  def testDefaultUniformRandomInitalizer(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      with self.session() as sess:
        with variable_scope.variable_scope("vs", use_resource=True):
          i = init_ops.random_uniform_initializer()
          z = variable_scope.get_variable("z1",
                                          shape=[],
                                          dtype=np.float32,
                                          initializer=i)

        sess.run(variables.global_variables_initializer())
        o = sess.run(z)
        self.assertAllClose(o, 0.5, 0.5, 0.5)

  def testVariablesRemainResident(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):

          w = variable_scope.get_variable(
              "w",
              shape=[4, 2],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(
                  np.array([[1, 2], [3, 4], [5, 6], [7, 8]],
                           dtype=np.float32)))
          b = variable_scope.get_variable(
              "b",
              shape=[2],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(
                  np.array([2, 3], dtype=np.float32)))

        x = array_ops.placeholder(np.float32, shape=[1, 4])
        y = math_ops.matmul(x, w) + b

        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)

      report_json = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report_json.reset()

      sess.run([train, loss], {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      sess.run([train, loss], {x: np.array([[1, 2, 3, 4]], dtype=np.float32)})
      sess.run([train, loss], {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      sess.run([train, loss], {x: np.array([[1, 2, 3, 4]], dtype=np.float32)})
      sess.run([train, loss], {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})

      w_dl = "1.0"
      w_ul = "out_1.0"
      b_dl = "2.0"
      b_ul = "out_2.0"

      report_json.parse_log()

      # The initialization is constant, so there are no events generated on the
      # IPU.

      report_json.assert_host_to_device_event_names(
          [w_dl, b_dl],
          "Weights/biases should be downloaded once, and the input no times "
          "because it is streamed")

      report_json.assert_device_to_host_event_names(
          [],
          "Weights/biases should not be uploaded, and the loss is streamed")

      # Explicitly fetch the weights
      vw, vb = sess.run([w, b])

      self.assertAllClose(
          np.array([[-1.3, -0.3], [1.7, 2.7], [2.9, 3.9], [3.5, 4.5]],
                   dtype=np.float32),
          vw,
          rtol=1e-4)

      self.assertAllClose(np.array([1.5, 2.5], dtype=np.float32),
                          vb,
                          rtol=1e-4)

      report_json.parse_log()

      report_json.assert_host_to_device_event_names(
          [], "Weights/biases/inputs should not be downloaded at all")

      report_json.assert_device_to_host_event_names(
          [w_ul, b_ul],
          "Weights/biases should be uploaded once (explicitly fetched)")

  def testResourceCountsAreCorrect(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):
          w1 = variable_scope.get_variable(
              "w1",
              shape=[4, 2],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(
                  np.array([[1, 2], [3, 4], [5, 6], [7, 8]],
                           dtype=np.float32)))
          b1 = variable_scope.get_variable(
              "b1",
              shape=[2],
              dtype=np.float32,
              trainable=False,
              initializer=init_ops.constant_initializer(
                  np.array([2, 3], dtype=np.float32)))
          w2 = variable_scope.get_variable(
              "w2",
              shape=[2, 2],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(
                  np.array([[1, 2], [3, 4]], dtype=np.float32)))
          b2 = variable_scope.get_variable(
              "b2",
              shape=[2],
              dtype=np.float32,
              trainable=False,
              initializer=init_ops.constant_initializer(
                  np.array([2, 3], dtype=np.float32)))

        x = array_ops.placeholder(np.float32, shape=[1, 4])
        y = math_ops.matmul(x, w1) + b1
        y = math_ops.matmul(y, w2) + b2

        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)

      report_json = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report_json.reset()

      sess.run([train, loss], {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      sess.run([train, loss], {x: np.array([[1, 2, 3, 4]], dtype=np.float32)})
      sess.run([train, loss], {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      sess.run([train, loss], {x: np.array([[1, 2, 3, 4]], dtype=np.float32)})
      sess.run([train, loss], {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})

      w1_dl = "1.0"
      b1_dl = "2.0"
      w2_dl = "3.0"
      b2_dl = "4.0"

      # biases are not outputs of the graph
      w1_ul = "out_1.0"
      w2_ul = "out_2.0"

      report_json.parse_log()

      # The initialization is constant, so there are no events generated on the
      # IPU.

      report_json.assert_host_to_device_event_names(
          [w1_dl, b1_dl, w2_dl, b2_dl],
          "Weights/biases should be downloaded once, and the input no times "
          "because it is streamed")

      # Weights should not be uploaded, and the loss is streamed
      report_json.assert_device_to_host_event_names(
          [],
          "Weights/biases should not be uploaded, and the loss is streamed")

      # Explicitly fetch the first set of weights and biases
      vw, vb = sess.run([w1, b1])

      self.assertAllClose(np.array(
          [[100.00576782, 86.60944366], [57.62784195, 51.23856354],
           [93.45920563, 82.40240479], [155.36032104, 135.74447632]],
          dtype=np.float32),
                          vw,
                          rtol=1e-4)

      self.assertAllClose(np.array([2, 3], dtype=np.float32), vb, rtol=1e-4)

      report_json.parse_log()

      report_json.assert_host_to_device_event_names(
          [], "Weights/biases/inputs should not be downloaded at all")

      # Note all weights are fetched as a group
      report_json.assert_device_to_host_event_names(
          [w1_ul, w2_ul],
          "Weights/biases should be uploaded once (explicitly fetched)")

  def testTuplesOfTuplesAreStreamed(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):
          pa = array_ops.placeholder(np.int64, [2, 2], name="a")
          pb = array_ops.placeholder(np.int64, [2, 2], name="b")
          pc = array_ops.placeholder(np.int64, [2, 2], name="c")
          c = control_flow_ops.tuple((pa + pc, pb + pc))

      report_json = tu.ReportJSON(self, sess)

      report_json.reset()
      in0 = np.full((2, 2), 7)
      in1 = np.full((2, 2), 6)
      in2 = np.full((2, 2), 5)
      fd = {
          pa: in0,
          pb: in1,
          pc: in2,
      }
      out = sess.run(c, fd)
      self.assertEqual(len(out), 2)
      self.assertAllClose(out, (np.full((2, 2), 12), np.full((2, 2), 11)))

      report_json.parse_log()
      report_json.assert_host_to_device_event_names(
          [], "No io events implies the data was streamed")
      report_json.assert_device_to_host_event_names(
          [], "No io events implies the data was streamed")

  def testNonModifiedResourceIsNotOverwrittenInPlaceOp(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.configure_ipu_system()

    # This test verifies that if we have a resource varaible (w) which is marked
    # as not modified then a copy is inserted to make sure it is not overwritten
    # between executions if it is used by an inplace op
    w_val = [1, 2, 3, 4]
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):
          w = variable_scope.get_variable(
              "w",
              shape=[4],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(
                  np.array(w_val, dtype=np.float32)))

        px = array_ops.placeholder(np.float32, shape=[4])
        y = w + px

      report_json = tu.ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())

      report_json.reset()
      xs = [
          np.array([7, 3, 5, 9], dtype=np.float32),
          np.array([1, 8, 3, 4], dtype=np.float32),
          np.array([9, 2, 2, 6], dtype=np.float32)
      ]
      for x in xs:
        out = sess.run(y, {px: x})
        self.assertAllClose(out, x + w_val)

      report_json.parse_log()

      w_dl = "1.0"
      report_json.assert_host_to_device_event_names(
          [w_dl], "w should be copied to device once and "
          "that should be the only io event")
      report_json.assert_device_to_host_event_names(
          [], "w should be copied to device once and "
          "that should be the only io event")


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
