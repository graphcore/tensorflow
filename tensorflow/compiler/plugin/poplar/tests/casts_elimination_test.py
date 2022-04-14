# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import os
import numpy as np
import pva

from tensorflow.python.ipu import test_utils as tu
from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops


class IpuFuseOpsTest(xla_test.XLATestCase):
  def testReductionSumVectorF16NoConverts(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [4096], name="a")
        output = math_ops.reduce_sum(pa, axis=[0])

      fd = {pa: np.ones([4096])}
      result = sess.run(output, fd)
      self.assertAllClose(result, 4096)

    report = pva.openReport(report_helper.find_report())
    # Check that there are no casts to float at the beginning.
    ok = [
        'Sum/reduce*/ReduceOnTile/InToIntermediateNoExchange/Reduce',
        'Sum/reduce*/ReduceFinalStage/IntermediateToOutput/Reduce'
    ]
    self.assert_all_compute_sets_and_list(report, ok)

  def testNoCastsF32ToF16ToF32(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [3])
        b = math_ops.cast(pa, np.float16)
        c = math_ops.cast(b, np.float32)

      fd = {pa: [2.0, 0.5, 1.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [2.0, 0.5, 1.0])

    self.assert_num_reports(report_helper, 0)

  def testNoCastsF16ReduceWithReshape(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [3, 4])
        a = gen_array_ops.reshape(pa, [4, 3])
        a = math_ops.reduce_sum(a, axis=(1))

      fd = {pa: np.ones([3, 4])}
      result = sess.run(a, fd)
      self.assertAllClose(result, [3.0, 3.0, 3.0, 3.0])

    report = pva.openReport(report_helper.find_report())
    ok = [
        'Sum/reduce*/Reduce',
    ]
    self.assert_all_compute_sets_and_list(report, ok)

  def testMultipleReduces(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [3])
        pb = array_ops.placeholder(np.float16, [3])
        a = math_ops.cast(pa, np.float32)
        a = math_ops.reduce_sum(a)
        a = math_ops.cast(a, np.float16)
        b = math_ops.cast(pb, np.float32)
        b = math_ops.reduce_sum(b)
        b = math_ops.cast(b, np.float16)
        c = a + b

      fd = {pa: [2.0, 0.5, 1.0], pb: [1.0, 1.0, 2.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, 7.5)

    report = pva.openReport(report_helper.find_report())
    ok = ['Sum/reduce*/Reduce', 'Sum_1/reduce*/Reduce', 'add/add']
    self.assert_all_compute_sets_and_list(report, ok)

  def testNoCastsF16ToF32ToF16(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [3])
        b = math_ops.cast(pa, np.float32)
        c = math_ops.cast(b, np.float16)

      fd = {pa: [2.0, 0.5, 1.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [2.0, 0.5, 1.0])

    self.assert_num_reports(report_helper, 0)

  def testDontRemoveCastsIfUsed(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, [3])
        b = math_ops.cast(pa, np.float32)
        const = array_ops.constant(1.0, np.float32)
        b = b + const
        c = math_ops.cast(b, np.float16)

      fd = {pa: [2.0, 0.5, 1.0]}
      result = sess.run(c, fd)
      self.assertAllClose(result, [3.0, 1.5, 2.0])

    report = pva.openReport(report_helper.find_report())
    ok = ['expression/Cast', 'expression/Op/Add', 'convert.*/Cast']
    self.assert_all_compute_sets_and_list(report, ok)

  def testReduceMean(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      shape = [2, 10000]
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, shape)
        output = math_ops.reduce_mean(pa, axis=[1])

      val = np.finfo(np.float16).max / 2
      result = sess.run(output, {pa: np.full(shape, val)})
      self.assertAllClose(result, [val, val])

    report = pva.openReport(report_helper.find_report())
    ok = ['Mean/fusion/Reduce', 'Mean/convert*/Cast']
    self.assert_all_compute_sets_and_list(report, ok)

  def testReduceMax(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      shape = [2, 10000]
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float16, shape)
        a = math_ops.cast(pa, np.float32)
        output = math_ops.reduce_max(a, axis=[1])

      val = np.finfo(np.float16).max / 2
      result = sess.run(output, {pa: np.full(shape, val)})
      self.assertAllClose(result, [val, val])

    report = pva.openReport(report_helper.find_report())
    ok = ['Max/reduce*/Reduce', 'Cast/convert*/Cast']
    self.assert_all_compute_sets_and_list(report, ok)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=2 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
