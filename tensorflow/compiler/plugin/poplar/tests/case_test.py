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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.ipu import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.eager import function as eager_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import googletest


class CaseTest(xla_test.XLATestCase):
  def testCaseSimple(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def my_graph(pa, pb, pc):
        with ipu.scopes.ipu_scope("/device:IPU:0"):

          @eager_function.defun
          def b0(x, y):
            return x + y

          @eager_function.defun
          def b1(x, y):
            return x - y

          @eager_function.defun
          def b2(x, y):
            return x * y

          branches = [
              f.get_concrete_function(array_ops.zeros_like(pb),
                                      array_ops.zeros_like(pc))
              for f in [b0, b1, b2]
          ]

          c_out = gen_functional_ops.case(pa,
                                          input=[pb, pc],
                                          Tout=[dtypes.float32],
                                          branches=branches)

          return [c_out[0]]

      with ops.device('cpu'):
        pa = array_ops.placeholder(np.int32, [], name="a")
        pb = array_ops.placeholder(np.float32, [2], name="b")
        pc = array_ops.placeholder(np.float32, [2], name="c")

      out = ipu.ipu_compiler.compile(my_graph, [pa, pb, pc])

      result = sess.run(out, {pa: 0, pb: [0., 1.], pc: [1., 5.]})
      self.assertAllClose(result[0], [1., 6.])

      result = sess.run(out, {pa: 1, pb: [0., 1.], pc: [1., 5.]})
      self.assertAllClose(result[0], [-1., -4.])

      result = sess.run(out, {pa: 2, pb: [0., 1.], pc: [1., 5.]})
      self.assertAllClose(result[0], [0., 5.])

      result = sess.run(out, {pa: 10, pb: [0., 1.], pc: [1., 5.]})
      self.assertAllClose(result[0], [0., 5.])

      self.assert_num_reports(report_helper, 1)

  def testCaseVariables(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def my_graph(pa, pb):
        with ipu.scopes.ipu_scope("/device:IPU:0"):

          @eager_function.defun
          def b0(x, y):
            return x + y

          @eager_function.defun
          def b1(x, y):
            return x - y

          @eager_function.defun
          def b2(x, y):
            return x * y

          v = variable_scope.get_variable('b0',
                                          dtype=dtypes.float32,
                                          initializer=[1., 5.])

          branches = [
              f.get_concrete_function(array_ops.zeros_like(pb),
                                      array_ops.zeros_like(v))
              for f in [b0, b1, b2]
          ]

          c_out = gen_functional_ops.case(pa,
                                          input=[pb, v],
                                          Tout=[dtypes.float32],
                                          branches=branches)

          return [c_out[0]]

      with ops.device('cpu'):
        pa = array_ops.placeholder(np.int32, [], name="a")
        pb = array_ops.placeholder(np.float32, [2], name="b")

      out = ipu.ipu_compiler.compile(my_graph, [pa, pb])

      sess.run(variables_lib.global_variables_initializer())

      result = sess.run(out, {pa: 0, pb: [0., 1.]})
      self.assertAllClose(result[0], [1., 6.])

      result = sess.run(out, {pa: 1, pb: [0., 1.]})
      self.assertAllClose(result[0], [-1., -4.])

      result = sess.run(out, {pa: 2, pb: [0., 1.]})
      self.assertAllClose(result[0], [0., 5.])

      result = sess.run(out, {pa: 10, pb: [0., 1.]})
      self.assertAllClose(result[0], [0., 5.])

      self.assert_num_reports(report_helper, 1)


if __name__ == "__main__":
  googletest.main()
