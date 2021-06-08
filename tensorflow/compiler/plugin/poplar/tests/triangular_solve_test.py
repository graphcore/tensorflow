# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.framework import ops
from tensorflow.python.platform import googletest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import variables


class TriangularSolvePerformanceTest(xla_test.XLATestCase):
  # Overriding abstract method.
  def cached_session(self):
    return 0

  # Overriding abstract method.
  def test_session(self):
    return 0

  def _solveTestImpl(self, n, m, block_size, lower, adjoint):
    with self.session() as sess:
      rng = np.random.RandomState(0)
      a = np.tril(rng.rand(n, n) - 0.5) / (2.0 * n) + np.eye(n)
      if not lower:
        a = np.swapaxes(a, -1, -2)
      b = rng.randn(n, m)

      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, a.shape, name="a")
        pb = array_ops.placeholder(np.float32, b.shape, name="b")
        px = linalg_ops.matrix_triangular_solve(pa,
                                                pb,
                                                lower=lower,
                                                adjoint=adjoint,
                                                name="x")

      report = tu.ReportJSON(self,
                             sess,
                             triangular_solve_expander_block_size=block_size)

      sess.run(variables.global_variables_initializer())

      report.reset()
      sess.run(px, {pa: a, pb: b})

      report.parse_log(assert_len=4)

      return report

  def testLowerAdjoint(self):
    report = self._solveTestImpl(64, 64, 16, True, True)
    report.assert_num_execution_reports_equal(1)
    report.assert_execution_report_cycles(0, 497693, tolerance=0.1)
    report.assert_max_tile_memory(2907, tolerance=0.1)
    report.assert_total_tile_memory(423760, tolerance=0.1)

  def testLowerNonAdjoint(self):
    report = self._solveTestImpl(64, 64, 16, True, False)
    report.assert_num_execution_reports_equal(1)
    report.assert_execution_report_cycles(0, 500080, tolerance=0.1)
    report.assert_max_tile_memory(3240, tolerance=0.1)
    report.assert_total_tile_memory(426638, tolerance=0.1)

  def testUpperAdjoint(self):
    report = self._solveTestImpl(64, 64, 16, False, True)
    report.assert_num_execution_reports_equal(1)
    report.assert_execution_report_cycles(0, 509228, tolerance=0.1)
    report.assert_max_tile_memory(3570, tolerance=0.1)
    report.assert_total_tile_memory(430342, tolerance=0.1)

  def testUpperNonAdjoint(self):
    report = self._solveTestImpl(64, 64, 16, False, False)
    report.assert_num_execution_reports_equal(1)
    report.assert_execution_report_cycles(0, 470171, tolerance=0.1)
    report.assert_max_tile_memory(2702, tolerance=0.1)
    report.assert_total_tile_memory(423944, tolerance=0.1)


if __name__ == "__main__":
  googletest.main()
