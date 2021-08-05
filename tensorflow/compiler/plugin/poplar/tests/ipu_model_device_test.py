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

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.platform import googletest
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class IpuIpuModelTest(xla_test.XLATestCase):
  def testIpuModelDevice(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        output = pa + pb

      cfg = IPUConfig()
      cfg.configure_ipu_system()

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      result = sess.run(output, fd)
      self.assertAllClose(result, [[1., 2.], [6., 8.]])

  def testIpuModelDeviceWithNoReport(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        output = pa + pb

      with ops.device('cpu'):
        with ops.control_dependencies([output]):
          report = gen_ipu_ops.ipu_event_trace()

      cfg = IPUConfig()
      cfg.configure_ipu_system()

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      sess.run(report, fd)

      result = sess.run(output, fd)
      rep = sess.run(report, fd)
      self.assertAllClose(result, [[1., 2.], [6., 8.]])
      self.assertTrue(len(rep) == 0)

  def testIpuModelDeviceWithReport(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        output = pa + pb

      with ops.device('cpu'):
        with ops.control_dependencies([output]):
          report = gen_ipu_ops.ipu_event_trace()

      report_json = tu.ReportJSON(self, sess)

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      sess.run(report, fd)

      result = sess.run(output, fd)
      rep = sess.run(report, fd)
      self.assertAllClose(result, [[1., 2.], [6., 8.]])

      types = report_json.parse_events(rep, assert_len=4)
      self.assertEqual(1, types[IpuTraceEvent.COMPILE_BEGIN])
      self.assertEqual(1, types[IpuTraceEvent.COMPILE_END])
      self.assertEqual(1, types[IpuTraceEvent.EXECUTE])
      self.assertEqual(1, types[IpuTraceEvent.LOAD_ENGINE])

  def testIpuModelDeviceWithMultipleReport(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        out1 = pa + pb
        out2 = pa - pb

      with ops.device('cpu'):
        with ops.control_dependencies([out1, out2]):
          report = gen_ipu_ops.ipu_event_trace()

      cfg = IPUConfig()
      cfg.ipu_model.compile_ipu_code = False
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.configure_ipu_system()

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      sess.run(report, fd)

      result = sess.run(out1, fd)
      self.assertAllClose(result, [[1., 2.], [6., 8.]])

      result = sess.run(out2, fd)
      rep = sess.run(report, fd)

      self.assertAllClose(result, [[1., 0.], [-2., -2.]])

      # 2x engine, 2x compile_begin, 2x compile_end, 2x load engine
      self.assertEqual(len(rep), 8)

  def testEngineCompilationOptions(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [480], name="a")
        pb = array_ops.placeholder(np.float32, [480], name="b")
        output = pa + pb

      cfg = IPUConfig()
      cfg.compilation_poplar_options = {'some_option': 'some_value'}
      cfg.configure_ipu_system()

      fd = {pa: np.zeros([480]), pb: np.zeros([480])}
      with self.assertRaisesRegex(errors.InternalError,
                                  "invalid_option: Unrecognised"):
        sess.run(output, fd)

  def testNamedOperations(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        with ops.name_scope('my_ops'):
          out = math_ops.add(pa, pb, 'my_add_op')

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}

      result = sess.run(out, fd)
      self.assertAllClose(result, [[1., 2.], [6., 8.]])

    report = pva.openReport(report_helper.find_report())
    ok = ['__seed*', 'my_ops/my_add_op/add']
    self.assert_all_compute_sets_and_list(report, ok)


if __name__ == "__main__":
  googletest.main()
