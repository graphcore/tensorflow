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
import pva
from tensorflow.python.ipu import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import googletest
from tensorflow.python import ipu


class SubcomputationGraphCachingTest(xla_test.XLATestCase):
  @test_util.deprecated_graph_mode_only
  def testSimpleCaching(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def f_1(x):
        return math_ops.square(x, name="namef1")

      def f_cond(x1, z):
        cond_1 = control_flow_ops.cond(math_ops.less(z[0], z[1]),
                                       lambda: f_1(x1), lambda: f_1(x1))
        return cond_1

      with ops.device('cpu'):
        x1 = array_ops.placeholder(dtypes.int32, [2, 2])
        z = array_ops.placeholder(dtypes.int32, [2])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r1 = ipu.ipu_compiler.compile(f_cond, inputs=[x1, z])
        i_x1 = np.full((2, 2), 10)
        i_z = np.full((2), 8)

        sess.run(r1, {x1: i_x1, z: i_z})

      report = pva.openReport(report_helper.find_report())
      self.assert_compute_sets_matches(
          report, '*namef1*', 1, "There should be only one f_1 due to cash.")

  @test_util.deprecated_graph_mode_only
  def testNotSameFunctions(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    # f_1, f_2 are not the same
    with self.session() as sess:

      def f_1(x):
        return math_ops.square(x, name="namef1")

      def f_2(x):
        return 2 * math_ops.square(x, name="namef2")

      def f_cond(x1):
        cond_1 = control_flow_ops.cond(math_ops.less(1, 0), lambda: f_1(x1),
                                       lambda: f_1(x1))
        cond_2 = control_flow_ops.cond(math_ops.less(1, 0), lambda: f_2(x1),
                                       lambda: f_2(x1))

        return cond_1 + cond_2

      with ops.device('cpu'):
        x1 = array_ops.placeholder(dtypes.int32, [8, 2])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r1 = ipu.ipu_compiler.compile(f_cond, inputs=[x1])
        i_x1 = np.full((8, 2), 10)

        sess.run(r1, {x1: i_x1})

      report = pva.openReport(report_helper.find_report())
      self.assert_compute_sets_matches(
          report, '*namef1*', 1, "There should be only one f_1 due to cash.")

      self.assert_compute_sets_matches(
          report, '*namef2*', 1, "There should be only one f_2 due to cash.")

  @test_util.deprecated_graph_mode_only
  def testSameFunctions(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    # f_1, f_2 are the same
    with self.session() as sess:

      def f_1(x):
        return math_ops.square(x, name="namef1")

      def f_2(x):
        return math_ops.square(x, name="namef2")

      def f_cond(x1):
        cond_1 = control_flow_ops.cond(math_ops.less(1, 0), lambda: f_1(x1),
                                       lambda: f_1(x1))
        cond_2 = control_flow_ops.cond(math_ops.less(1, 0), lambda: f_2(x1),
                                       lambda: f_2(x1))

        return cond_1 + cond_2

      with ops.device('cpu'):
        x1 = array_ops.placeholder(dtypes.int32, [2, 2])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r1 = ipu.ipu_compiler.compile(f_cond, inputs=[x1])
        i_x1 = np.full((2, 2), 10)

        sess.run(r1, {x1: i_x1})

      report = pva.openReport(report_helper.find_report())
      self.assert_compute_sets_matches(
          report, '*namef1*', 1, "There should be only one f_1 due to cash.")

      self.assert_compute_sets_matches(
          report, '*namef2*', 0,
          "There should not be f_2, as it is the same as f_1, due to cash.")

  @test_util.deprecated_graph_mode_only
  def testWhenSideEffect(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def f_1(x):
        rand_num = 10 * random_ops.random_uniform(shape=[2, 2],
                                                  minval=1,
                                                  maxval=9,
                                                  dtype=dtypes.int32,
                                                  name="namef1")
        return rand_num * x

      def f_cond(x1, z):
        cond_1 = control_flow_ops.cond(math_ops.less(z[0], z[1]),
                                       lambda: f_1(x1), lambda: f_1(x1))
        return cond_1

      with ops.device('cpu'):
        x1 = array_ops.placeholder(dtypes.int32, [2, 2])
        z = array_ops.placeholder(dtypes.int32, [2])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r1 = ipu.ipu_compiler.compile(f_cond, inputs=[x1, z])
        i_x1 = np.full((2, 2), 10)
        i_z = np.full((2), 8)

        sess.run(r1, {x1: i_x1, z: i_z})

      report = pva.openReport(report_helper.find_report())
      self.assert_compute_sets_matches(
          report, '*namef1*', 2,
          "f1 should be on the list twice as it should not be cashed "
          "due to SideEffect.")


if __name__ == "__main__":
  googletest.main()
