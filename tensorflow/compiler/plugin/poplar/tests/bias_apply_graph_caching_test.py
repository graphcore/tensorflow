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
import pva
import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables


class BiasApplyGraphCachingTest(xla_test.XLATestCase):
  def testMatch(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        biases1 = array_ops.placeholder(np.float32, shape=[2])
        biases2 = array_ops.placeholder(np.float32, shape=[2])
        biases3 = array_ops.placeholder(np.float32, shape=[2])
        grads1 = array_ops.placeholder(np.float32, shape=[2, 10])
        grads2 = array_ops.placeholder(np.float32, shape=[2, 10])
        grads3 = array_ops.placeholder(np.float32, shape=[2, 10])
        vlr = array_ops.placeholder(np.float32, shape=[])

        def bias_apply(bias, grad, lr):
          return bias - math_ops.reduce_sum(grad, axis=1) * lr

        out = (bias_apply(biases1, grads1, vlr) +
               bias_apply(biases2, grads2, 0.1) +
               bias_apply(biases3, grads3, 0.2))

      sess.run(variables.global_variables_initializer())

      r = sess.run(
          out, {
              biases1: np.ones([2]),
              biases2: np.ones([2]),
              biases3: np.ones([2]),
              grads1: np.ones([2, 10]),
              grads2: np.ones([2, 10]),
              grads3: np.ones([2, 10]),
              vlr: 0.1
          })

    self.assertAllClose(r, [-1., -1.])
    report = pva.openReport(report_helper.find_report())
    self.assert_compute_sets_matches(report, "*ReduceOnTile*", 1)

  def testMatchBecauseEvenWhenNotInplace(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.configure_ipu_system()

    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        biases1 = array_ops.placeholder(np.float32, shape=[2])
        grads1 = array_ops.placeholder(np.float32, shape=[2, 10])
        grads2 = array_ops.placeholder(np.float32, shape=[2, 10])

        def bias_apply(bias, grad):
          return bias - math_ops.reduce_sum(grad, axis=1) * 0.1

        out = bias_apply(biases1, grads1) + bias_apply(biases1, grads2)

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      r = sess.run(
          out, {
              biases1: np.ones([2]),
              grads1: np.ones([2, 10]),
              grads2: np.ones([2, 10])
          })

    self.assertAllClose(r, [0., 0.])
    report = pva.openReport(report_helper.find_report())
    self.assert_compute_sets_matches(
        report, "*ReduceOnTile*", 1,
        "We should still reuse the code even though only one reduce is inplace"
    )


if __name__ == "__main__":
  googletest.main()
