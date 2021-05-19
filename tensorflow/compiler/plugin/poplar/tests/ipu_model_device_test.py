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

import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python.ipu import utils
from tensorflow.python.platform import googletest
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ipu.config import IPUConfig


class IpuIpuModelTest(xla_test.XLATestCase):
  def testIpuModelDevice(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        output = pa + pb

      opts = IPUConfig()
      opts._profiling.profiling = True  # pylint: disable=protected-access
      opts.configure_ipu_system()

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

      opts = IPUConfig()
      opts._profiling.profiling = False  # pylint: disable=protected-access
      opts.configure_ipu_system()

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      sess.run(report, fd)

      result = sess.run(output, fd)
      rep = sess.run(report, fd)
      self.assertAllClose(result, [[1., 2.], [6., 8.]])
      self.assertTrue(len(rep) == 0)

  def testIpuModelDeviceWithReport(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        output = pa + pb

      with ops.device('cpu'):
        with ops.control_dependencies([output]):
          report = gen_ipu_ops.ipu_event_trace()

      r = tu.ReportJSON(self, sess)

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      sess.run(report, fd)

      result = sess.run(output, fd)
      rep = sess.run(report, fd)
      self.assertAllClose(result, [[1., 2.], [6., 8.]])

      types = r.parse_events(rep, assert_len=4)
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

      opts = IPUConfig()
      opts._profiling.profiling = True  # pylint: disable=protected-access
      opts._profiling.profile_execution = True  # pylint: disable=protected-access
      opts.configure_ipu_system()

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

      opts = IPUConfig()
      opts.compilation_poplar_options = {'some_option': 'some_value'}
      opts.configure_ipu_system()

      fd = {pa: np.zeros([480]), pb: np.zeros([480])}
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "Unrecognised option"):
        sess.run(output, fd)

  def testNamedOperations(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        with ops.name_scope('my_ops'):
          out = math_ops.add(pa, pb, 'my_add_op')

      r = tu.ReportJSON(self, sess)

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      r.reset()

      result = sess.run(out, fd)
      self.assertAllClose(result, [[1., 2.], [6., 8.]])

      r.parse_log()

      ok = ['__seed*', 'my_ops/my_add_op/add']
      r.assert_all_compute_sets_and_list(ok)

  def testReportEveryNthExecution_FirstOnly(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        out = math_ops.add(pa, pb)

      r = tu.ReportJSON(self, sess)

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      r.reset()

      sess.run(out, fd)
      sess.run(out, fd)
      sess.run(out, fd)
      sess.run(out, fd)
      sess.run(out, fd)

      types = r.parse_log()
      self.assertEqual(types[IpuTraceEvent.EXECUTE], 5)
      self.assertEqual(
          len(r.get_execution_reports()), 1,
          "Only the first execution should have generated a report")

  def testReportEveryNthExecution_Every2(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        out = math_ops.add(pa, pb)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

      opts = IPUConfig()
      opts._profiling.profiling = True  # pylint: disable=protected-access
      opts._profiling.profile_execution = True  # pylint: disable=protected-access
      opts._profiling.report_every_nth_execution = 2  # pylint: disable=protected-access
      opts._profiling.use_poplar_text_report = False  # pylint: disable=protected-access
      opts.configure_ipu_system()

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      sess.run(report, fd)

      sess.run(out, fd)
      sess.run(out, fd)
      sess.run(out, fd)
      sess.run(out, fd)
      sess.run(out, fd)

      rep = sess.run(report, fd)
      r = tu.ReportJSON(self)
      types = r.parse_events(rep)
      self.assertEqual(types[IpuTraceEvent.EXECUTE], 5)
      self.assertEqual(
          len(r.get_execution_reports()), 3,
          "The 1st, 3rd and 5th execution should have generated a report")

  def testReportEveryNthExecution_Every1(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        out = math_ops.add(pa, pb)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

      opts = IPUConfig()
      opts._profiling.profiling = True  # pylint: disable=protected-access
      opts._profiling.profile_execution = True  # pylint: disable=protected-access
      opts._profiling.report_every_nth_execution = 1  # pylint: disable=protected-access
      opts._profiling.use_poplar_text_report = False  # pylint: disable=protected-access
      opts.configure_ipu_system()

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      sess.run(report, fd)

      sess.run(out, fd)
      sess.run(out, fd)
      sess.run(out, fd)
      sess.run(out, fd)
      sess.run(out, fd)

      rep = sess.run(report, fd)
      r = tu.ReportJSON(self)
      types = r.parse_events(rep)
      self.assertEqual(types[IpuTraceEvent.EXECUTE], 5)
      self.assertEqual(len(r.get_execution_reports()), 5,
                       "Every execution should have generated a report")

  def testJsonReport(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        out = math_ops.add(pa, pb)

      r = tu.ReportJSON(self, sess)

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      r.reset()

      sess.run(out, fd)

      r.parse_log(assert_len=4, assert_msg="engine, begin, end, execute")

  def testCborReport(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        out = math_ops.add(pa, pb)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

      opts = IPUConfig()
      opts._profiling.profiling = True  # pylint: disable=protected-access
      opts._profiling.profile_execution = True  # pylint: disable=protected-access
      opts._profiling.use_poplar_text_report = False  # pylint: disable=protected-access
      opts._profiling.use_poplar_cbor_report = True  # pylint: disable=protected-access
      opts.configure_ipu_system()

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      sess.run(report, fd)

      sess.run(out, fd)

      rep = sess.run(report, fd)
      evts = utils.extract_all_events(rep)
      self.assertEqual(len(evts), 4)  # engine, begin, end, execute

      self.assertEqual(evts[1].compile_end.compilation_report[0],
                       bytes(bytearray([217]))[0])
      self.assertEqual(evts[3].execute.execution_report[0],
                       bytes(bytearray([217]))[0])

  def testIpuEventsWithoutPoplarReporting(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        out = math_ops.add(pa, pb)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

      opts = IPUConfig()
      opts._profiling.profiling = False  # pylint: disable=protected-access
      opts._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      opts.configure_ipu_system()

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      sess.run(report, fd)

      sess.run(out, fd)

      rep = sess.run(report, fd)
      evts = utils.extract_all_events(rep)
      self.assertEqual(len(evts), 3)  # compile begin, compile end, execute

      for e in evts:
        if e.type == IpuTraceEvent.COMPILE_END:
          self.assertFalse(e.compile_end.compilation_report)
        if e.type == IpuTraceEvent.EXECUTE:
          self.assertFalse(e.execute.execution_report)

      sess.close()


if __name__ == "__main__":
  googletest.main()
