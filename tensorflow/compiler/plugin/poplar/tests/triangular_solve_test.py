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
import pva
from tensorflow.python.ipu import test_utils as tu

from tensorflow.compiler.tests import xla_test

from tensorflow.python.framework import ops
from tensorflow.python.platform import googletest
from tensorflow.python.ipu.config import IPUConfig
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
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 4
    cfg.optimizations.triangular_solve_expander_block_size = block_size
    cfg.configure_ipu_system()

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

      sess.run(variables.global_variables_initializer())

      sess.run(px, {pa: a, pb: b})

    return report_helper

  def testLowerAdjoint(self):
    report_helper = self._solveTestImpl(64, 64, 16, True, True)
    report = pva.openReport(report_helper.find_report())
    self.assert_execution_report_cycles(report, 1110077, tolerance=0.1)
    self.assert_max_tile_memory(report, 34526, tolerance=0.1)
    self.assert_total_tile_memory(report, 130713, tolerance=0.1)

  def testLowerNonAdjoint(self):
    report_helper = self._solveTestImpl(64, 64, 16, True, False)
    report = pva.openReport(report_helper.find_report())
    self.assert_execution_report_cycles(report, 1098318, tolerance=0.1)
    self.assert_max_tile_memory(report, 34703, tolerance=0.1)
    self.assert_total_tile_memory(report, 127188, tolerance=0.1)

  def testUpperAdjoint(self):
    report_helper = self._solveTestImpl(64, 64, 16, False, True)
    report = pva.openReport(report_helper.find_report())
    self.assert_execution_report_cycles(report, 1155219, tolerance=0.1)
    self.assert_max_tile_memory(report, 34001, tolerance=0.1)
    self.assert_total_tile_memory(report, 148162, tolerance=0.1)

  def testUpperNonAdjoint(self):
    report_helper = self._solveTestImpl(64, 64, 16, False, False)
    report = pva.openReport(report_helper.find_report())
    self.assert_execution_report_cycles(report, 1098082, tolerance=0.1)
    self.assert_max_tile_memory(report, 34149, tolerance=0.1)
    self.assert_total_tile_memory(report, 126310, tolerance=0.1)


if __name__ == "__main__":
  googletest.main()
