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
import pva
import numpy as np
import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.compiler.xla import xla
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn


class UpdateOpDependenciesTest(xla_test.XLATestCase):
  def testDontOutlineInplaceExpression(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [])
        pb = array_ops.placeholder(np.float32, [])
        pc = array_ops.placeholder(np.float32, [])
        pd = array_ops.placeholder(np.float32, [])
        e = pa + pb - pc + pd

      fd = {pa: 1, pb: 2, pc: 3, pd: 4}
      result = sess.run(e, fd)
      self.assertAllClose(result, 4)

    self.assert_num_reports(report_helper, 0)

  def testInplaceAddCopyWithInplacePeer(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      data_a = np.array([[10, -20], [5, 1]])
      data_b = np.array([[-12, 11], [12, -13]])
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2])
        pb = array_ops.placeholder(np.float32, [2, 2])
        c = array_ops.transpose(pa)
        d = pa + pb
        e = c / d

      fd = {
          pa: data_a,
          pb: data_b,
      }
      result = sess.run(e, fd)
      np_result = np.transpose(data_a) / (data_a + data_b)
      self.assertAllClose(result, np_result)

    report = pva.openReport(report_helper.find_report())
    ok = [
        'Copy_XLA_Args/*_to_transpose/transpose', 'add/add.*/Op/Add',
        'truediv/divide.*/Op/Divide'
    ]
    self.assert_all_compute_sets_and_list(report, ok)

  def testInplaceAddCopyWithInplacePeer2(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      data_a = np.array([[10, -10], [-5, 5]])
      data_b = np.array([[-15, 15], [25, -25]])
      data_c = 2
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2])
        pb = array_ops.placeholder(np.float32, [2, 2])
        pc = array_ops.placeholder(np.float32, [])
        a = array_ops.transpose(pa)
        b = pa + pb * pc
        c = a * pb + pc
        d = b / c

      fd = {
          pa: data_a,
          pb: data_b,
          pc: data_c,
      }
      np_result = (data_a + data_b * data_c) / (np.transpose(data_a) * data_b +
                                                data_c)
      result = sess.run(d, fd)
      self.assertAllClose(result, np_result)

    report = pva.openReport(report_helper.find_report())
    ok = [
        'add_1/add.*/expression/Op/Add', 'add_1/add.*/expression/Op/Multiply',
        'add/scaled-inplace-xb-y/AddTo', 'truediv/divide*/Op/Divide'
    ]
    self.assert_all_compute_sets_and_list(report, ok)

  def testInplaceOpAddCopyWithInplaceParent(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [3])
        pb = array_ops.placeholder(np.float32, [3])
        pc = array_ops.placeholder(np.float32, [])
        c = array_ops.slice(pa, [0], [2])
        d = array_ops.slice(pb, [0], [2])
        e = c + d
        f = e / pc
        g = array_ops.slice(pa, [1], [2])
        h = f + g

      fd = {
          pa: [1, 2, 3],
          pb: [5, 6, 7],
          pc: 2,
      }
      result = sess.run(h, fd)
      self.assertAllClose(result, [5, 7])

    report = pva.openReport(report_helper.find_report())
    ok = [
        'truediv/*/expression/Op/Add', 'truediv/*/expression/Op/Divide',
        'add_1/add.*/Add', 'host-exchange-local-copy'
    ]
    self.assert_all_compute_sets_and_list(report, ok)

  def testInplaceTuple(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def my_net(x):
        def cond(i, x, y):
          del x
          del y
          return i < 2

        def body(i, x, y):
          i = i + 1
          x = nn.tanh(x)
          y = nn.tanh(y)
          return (i, x, y)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, x, x), name='')[1:]

      with ops.device('cpu'):
        x = array_ops.placeholder(np.float32, [4])

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[x])

      x, y = sess.run(r, {x: np.full([4], 2)})
      self.assertAllClose(x, np.full([4], np.tanh(np.tanh(2))))
      self.assertAllClose(y, np.full([4], np.tanh(np.tanh(2))))

    report = pva.openReport(report_helper.find_report())
    ok = ['Copy_*_to_*', 'Tanh/tanh*/Op/Tanh', 'Tanh_1/tanh*/Op/Tanh']
    self.assert_all_compute_sets_and_list(report, ok)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
