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
# =============================================================================

import numpy as np
import pva

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python import ipu
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.ipu import test_utils as tu


def gelu_ref(features):
  retval = 0.5 * features * (1.0 + math_ops.erf(
      features / math_ops.cast(1.4142135623730951, features.dtype)))
  return retval


class GeluTest(test_util.TensorFlowTestCase):
  configured = False

  def __configureIPU(self, report_helper):
    if not self.configured:
      cfg = ipu.config.IPUConfig()
      cfg.ipu_model.compile_ipu_code = False
      cfg.auto_select_ipus = 1
      report_helper.set_autoreport_options(cfg)
      cfg.configure_ipu_system()
      self.configured = True

  def run_gelu_test(self, n, dtype):
    with self.session() as sess:
      report_helper = tu.ReportHelper()
      self.__configureIPU(report_helper)

      i_h = np.linspace(-10, 10, n, dtype=dtype)
      ref_h = gelu_ref(i_h)

      with ops.device("/device:IPU:0"):
        i = array_ops.placeholder(dtype, shape=[n])
        o = gelu_ref(i)

        test_h = sess.run(o, {i: i_h})

        self.assertAllCloseAccordingToType(ref_h, test_h)
        report = pva.openReport(report_helper.find_report())
        self.assert_compute_sets_contain_list(report,
                                              ["*/gelu-erf/Op/GeluErf"])

  @test_util.deprecated_graph_mode_only
  def testFP32(self):
    self.run_gelu_test(1000, np.float32)

  @test_util.deprecated_graph_mode_only
  def testFP16(self):
    self.run_gelu_test(1000, np.float16)


if __name__ == "__main__":
  googletest.main()
