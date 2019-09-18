# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
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

      tu.configure_ipu_system()

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

      tu.configure_ipu_system(False, False, False)

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      sess.run(report, fd)

      result, rep = sess.run([output, report], fd)
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

      tu.configure_ipu_system(text_report=False)

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      sess.run(report, fd)

      result, rep = sess.run([output, report], fd)
      self.assertAllClose(result, [[1., 2.], [6., 8.]])

      r = tu.ReportJSON(self)
      types = r.parse_events(rep, assert_len=3)
      self.assertEqual(1, types[IpuTraceEvent.COMPILE_BEGIN])
      self.assertEqual(1, types[IpuTraceEvent.COMPILE_END])
      self.assertEqual(1, types[IpuTraceEvent.EXECUTE])

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

      tu.configure_ipu_system()

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      sess.run(report, fd)

      result = sess.run(out1, fd)
      self.assertAllClose(result, [[1., 2.], [6., 8.]])

      result, rep = sess.run([out2, report], fd)
      self.assertAllClose(result, [[1., 0.], [-2., -2.]])

      # 2x compile_begin, 2x compile_end, 2x load engine
      self.assertEqual(len(rep), 6)

  def testEngineCompilationOptions(self):
    try:
      with self.session() as sess:
        with ops.device("/device:IPU:0"):
          pa = array_ops.placeholder(np.float32, [480], name="a")
          pb = array_ops.placeholder(np.float32, [480], name="b")
          output = pa + pb

        tu.configure_ipu_system(True,
                                True,
                                True,
                                engine_opts={"some_option": "some_value"})

        fd = {pa: np.zeros([480]), pb: np.zeros([480])}
        sess.run(output, fd)

        self.assertTrue(False)
    except errors.InvalidArgumentError:
      pass

  def testNamedOperations(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        with ops.name_scope('my_ops'):
          out = math_ops.add(pa, pb, 'my_add_op')

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(text_report=False)

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      sess.run(report, fd)

      result = sess.run(out, fd)
      self.assertAllClose(result, [[1., 2.], [6., 8.]])

      rep = sess.run(report, fd)
      r = tu.ReportJSON(self)
      r.parse_events(rep)

      ok = ['__seed*', 'my_ops/my_add_op/add']
      r.assert_all_compute_sets_and_list(ok)

  def testReportEveryNthExecution_FirstOnly(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        out = math_ops.add(pa, pb)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(compilation_trace=False, text_report=False)

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

      tu.configure_ipu_system(compilation_trace=False,
                              report_every_nth_execution=2,
                              text_report=False)

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

      tu.configure_ipu_system(compilation_trace=False,
                              report_every_nth_execution=1,
                              text_report=False)

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

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(text_report=False)

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      sess.run(report, fd)

      sess.run(out, fd)

      rep = sess.run(report, fd)
      r = tu.ReportJSON(self)
      r.parse_events(rep, assert_len=3, assert_msg="begin, end, execute")

  def testCborReport(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        out = math_ops.add(pa, pb)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(text_report=False, cbor_report=True)

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      sess.run(report, fd)

      sess.run(out, fd)

      rep = sess.run(report, fd)
      evts = tu.extract_all_events(rep)
      self.assertEqual(len(evts), 3)  # begin, end, execute

      self.assertEqual(evts[1].compile_end.compilation_report[0],
                       bytes(bytearray([217]))[0])
      self.assertEqual(evts[2].execute.execution_report[0],
                       bytes(bytearray([217]))[0])

  def testIpuEventsWithoutPoplarReporting(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        out = math_ops.add(pa, pb)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(enable_ipu_events=True,
                              compilation_trace=False,
                              io_trace=False,
                              execution_trace=False)

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      sess.run(report, fd)

      sess.run(out, fd)

      rep = sess.run(report, fd)
      evts = tu.extract_all_events(rep)
      self.assertEqual(len(evts), 3)  # compile begin, compile end, execute

      for e in evts:
        if e.type == IpuTraceEvent.COMPILE_END:
          self.assertFalse(e.compile_end.compilation_report)
        if e.type == IpuTraceEvent.EXECUTE:
          self.assertFalse(e.execute.execution_report)

      sess.close()


if __name__ == "__main__":
  googletest.main()
