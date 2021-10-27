# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import layers
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import googletest
import test_utils as tu


class ScalarElementWiseGraphTest(xla_test.XLATestCase):
  # Overriding abstract method.
  def cached_session(self):
    return 0

  # Overriding abstract method.
  def test_session(self):
    return 0

  def testDoNotCompileScalarElementWiseGraphWithParameter(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def my_graph(a, b):
        with ops.device("/device:IPU:0"):
          x = math_ops.add(a, b)
        return x

      with ops.device('cpu'):
        a = array_ops.placeholder(np.int32, name="a")
        b = array_ops.placeholder(np.int32, name="b")

      out = ipu.ipu_compiler.compile(my_graph, [a, b])

      fd = {a: np.int32(2), b: np.int32(3)}
      result = sess.run(out, fd)

      self.assert_num_reports(report_helper, 0)

      self.assertAllClose(result, [5])

  def testDoNotCompileScalarConstGraph(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def my_graph(a, b):
        with ops.device("/device:IPU:0"):
          x = math_ops.add(a, b)
        return x

      with ops.device('cpu'):
        a = 2
        b = 3
      out = ipu.ipu_compiler.compile(my_graph, [a, b])

      result = sess.run(out)

      # If compile was called, a report_json would be generated
      self.assert_num_reports(report_helper, 0)

      self.assertEqual(result, [5])

  def testDoNotCompileScalarElementWiseGraphWithParameterAdd1(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def my_graph(a, b):
        with ops.device("/device:IPU:0"):
          x = math_ops.add(a, b)
          x = math_ops.add(x, 1)
        return x

      with ops.device('cpu'):
        a = array_ops.placeholder(np.int32, name="a")
        b = array_ops.placeholder(np.int32, name="b")

      out = ipu.ipu_compiler.compile(my_graph, [a, b])

      fd = {a: np.int32(2.0), b: np.int32(3.0)}
      result = sess.run(out, fd)

      # If compile was called, a report_json would be generated
      self.assert_num_reports(report_helper, 0)

      self.assertEqual(result, [6])

  @test_util.deprecated_graph_mode_only
  def testWhenSomeScalarOnDevice(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.configure_ipu_system()

    def conv(x, ksize, stride, filters_out):
      return layers.Conv2D(
          filters_out,
          ksize,
          stride,
          'SAME',
          kernel_initializer=init_ops.constant_initializer(0.1),
          bias_initializer=init_ops.constant_initializer(0.0))(x)

    def graph1(x):
      x = conv(x, 3, 1, 2)
      x = math_ops.reduce_mean(x)
      x = array_ops.reshape(x, [])

      with variable_scope.variable_scope("vs",
                                         use_resource=True,
                                         reuse=variable_scope.AUTO_REUSE):
        z = variable_scope.get_variable(
            "var",
            shape=[],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(1.0))
      x = x + z

      z = state_ops.assign_add(z, x)
      return x

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    def graph2():
      with variable_scope.variable_scope("vs",
                                         use_resource=True,
                                         reuse=variable_scope.AUTO_REUSE):
        z = variable_scope.get_variable(
            "var",
            shape=[],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(1.0))
      return state_ops.assign_add(z, 1.0)

    with ops.device("/device:IPU:0"):
      output1 = ipu_compiler.compile(graph1, inputs=[x])
      output2 = ipu_compiler.compile(graph2, inputs=[])

    tu.move_variable_initialization_to_cpu()

    with tu.ipu_session() as sess:

      report_json = tu.ReportJSON(self, sess)
      report_json.reset()
      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      result1 = sess.run(output1, {x: np.ones(x.shape)})
      report_json.parse_log()
      report_json.assert_contains_host_to_device_transfer_event()
      report_json.reset()
      self.assert_num_reports(report_helper, 1)
      report_helper.clear_reports()

      result2 = sess.run(output2)
      report_json.parse_log()
      # Check that there was a copy from device to host. If there was no the copy
      # there would be one compile event at this place. We see no compile event
      # as expected.
      report_json.assert_contains_device_to_host_transfer_event()
      report_json.reset()
      self.assert_num_reports(report_helper, 0)

      result3 = sess.run(output1, {x: np.ones(x.shape)})
      report_json.parse_log()
      report_json.assert_contains_host_to_device_transfer_event()
      report_json.reset()
      self.assert_num_reports(report_helper, 0)

      # Read comment for case result2.
      result4 = sess.run(output2)
      report_json.parse_log()
      report_json.assert_contains_device_to_host_transfer_event()
      report_json.reset()
      self.assert_num_reports(report_helper, 0)

      self.assertAllClose(result1, [2.25])
      self.assertAllClose(result2, [4.25])
      self.assertAllClose(result3, [5.5])
      self.assertAllClose(result4, [10.75])


if __name__ == "__main__":
  googletest.main()
