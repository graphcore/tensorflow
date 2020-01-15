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
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.plugin.poplar.tests.test_utils import ReportJSON
from tensorflow.python import ipu
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class ScalarElementWiseGraphTest(xla_test.XLATestCase):
  def testDoNotCompileScalarElementWiseGraphWithParameter(self):
    with self.session() as sess:

      def my_graph(a, b):
        with ops.device("/device:IPU:0"):
          x = math_ops.add(a, b)
        return x

      with ops.device('cpu'):
        a = array_ops.placeholder(np.int32, name="a")
        b = array_ops.placeholder(np.int32, name="b")

      out = ipu.ipu_compiler.compile(my_graph, [a, b])
      report = ReportJSON(self, sess, device_count_override=1)
      report.reset()

      fd = {a: np.int32(2), b: np.int32(3)}
      result = sess.run(out, fd)

      report.parse_log()
      report.assert_contains_no_compile_event()

      self.assertAllClose(result, [5])

  def testDoNotCompileScalarConstGraph(self):
    with self.session() as sess:

      def my_graph(a, b):
        with ops.device("/device:IPU:0"):
          x = math_ops.add(a, b)
        return x

      with ops.device('cpu'):
        a = 2
        b = 3
      out = ipu.ipu_compiler.compile(my_graph, [a, b])
      report = ReportJSON(self, sess, device_count_override=1)
      report.reset()

      result = sess.run(out)

      report.parse_log()
      report.assert_contains_no_compile_event()

      self.assertEqual(result, [5])

  def testDoNotCompileScalarElementWiseGraphWithParameterAdd1(self):
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
      report = ReportJSON(self, sess, device_count_override=1)
      report.reset()

      fd = {a: np.int32(2.0), b: np.int32(3.0)}
      result = sess.run(out, fd)

      report.parse_log()
      report.assert_contains_no_compile_event()

      self.assertEqual(result, [6])

  def testSomeTensorsOnDeviceSomeOnHostForScalarElementWiseGrap(self):
    with self.session() as sess:

if __name__ == "__main__":
  googletest.main()
