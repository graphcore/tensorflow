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
import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.compiler.xla import xla
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn


class UpdateOpDependenciesTest(xla_test.XLATestCase):
  def testDontOutlineInplaceExpression(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [])
        pb = array_ops.placeholder(np.float32, [])
        pc = array_ops.placeholder(np.float32, [])
        pd = array_ops.placeholder(np.float32, [])
        e = pa + pb - pc + pd

      report = tu.ReportJSON(self, sess)

      report.reset()
      fd = {pa: 1, pb: 2, pc: 3, pd: 4}
      result = sess.run(e, fd)
      self.assertAllClose(result, 4)

      report.parse_log()
      report.assert_contains_no_compile_event()

  def tesInplaceAddCopyWithInplacePeer(self):
    with self.session() as sess:
      data_a = np.array([[10, -20], [5, 1]])
      data_b = np.array([[-12, 11], [12, -13]])
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, [2, 2])
        pb = array_ops.placeholder(np.float32, [2, 2])
        c = array_ops.transpose(pa)
        d = pa + pb
        e = c / d

      report = tu.ReportJSON(self, sess)
      report.reset()

      fd = {
          pa: data_a,
          pb: data_b,
      }
      result = sess.run(e, fd)
      np_result = np.transpose(data_a) / (data_a + data_b)
      self.assertAllClose(result, np_result)

      report.parse_log(
          assert_len=4,
          assert_msg="engine, compile_begin, compile_end, execute")

      ok = [
          '__seed*', 'host-exchange-local-copy-',
          'Copy_XLA_Args/arg0.*_to_transpose/transpose', 'add/add.*/AddTo',
          'truediv/divide.*/Op/Divide'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def tesInplaceAddCopyWithInplacePeer2(self):
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

      report = tu.ReportJSON(self, sess)
      report.reset()
      fd = {
          pa: data_a,
          pb: data_b,
          pc: data_c,
      }
      np_result = (data_a + data_b * data_c) / (np.transpose(data_a) * data_b +
                                                data_c)
      result = sess.run(d, fd)
      self.assertAllClose(result, np_result)

      report.parse_log(
          assert_len=4,
          assert_msg="engine, compile_begin, compile_end, execute")

      ok = [
          '__seed*', 'Copy_XLA_Args/arg0.*_to_transpose/transpose'
          'mul/multiply.*/Op/Multiply', 'add/add.*/AddTo',
          'mul_1/multiply.*/Op/Multiply', 'add_1/add.*/AddTo',
          'truediv/divide.*/Op/Divide'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testInplaceOpAddCopyWithInplaceParent(self):
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

      report = tu.ReportJSON(self, sess)
      report.reset()
      fd = {
          pa: [1, 2, 3],
          pb: [5, 6, 7],
          pc: 2,
      }
      result = sess.run(h, fd)
      self.assertAllClose(result, [5, 7])

      report.parse_log(
          assert_len=4,
          assert_msg="engine, compile_begin, compile_end, execute")

      ok = [
          '__seed*', 'truediv/*/expression/Op/Add',
          'truediv/*/expression/Op/Divide', 'add_1/add.*/Add',
          'host-exchange-local-copy'
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testInplaceTuple(self):
    with self.session() as sess:

      def my_net(x):
        def cond(i, x, y):
          del x
          del y
          return i < 1

        def body(i, x, y):
          i = i + 1
          x = nn.tanh(x)
          y = nn.tanh(y)
          return (i, x, y)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, x, x), name='')[1:]

      with ops.device('cpu'):
        x = array_ops.placeholder(np.float32, [4])

      report = tu.ReportJSON(self, sess)

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[x])

      report.reset()
      x, y = sess.run(r, {x: np.full([4], 2)})
      self.assertAllClose(x, np.full([4], np.tanh(2)))
      self.assertAllClose(y, np.full([4], np.tanh(2)))

      report.parse_log(assert_len=4)

      ok = [
          '__seed*', 'Copy_*_to_*', 'Tanh/tanh*/Op/Tanh',
          'Tanh_1/tanh*/Op/Tanh'
      ]
      report.assert_all_compute_sets_and_list(ok)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
