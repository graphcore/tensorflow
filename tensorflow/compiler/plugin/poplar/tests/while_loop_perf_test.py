# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.python import ipu
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class WhileLoopPerfTest(xla_test.XLATestCase):
  def testIpuWhilePerfTest(self):
    cfg = ipu.config.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    with self.session() as sess:

      def cond(i, v):
        del v
        return math_ops.less(i, 15)

      def body(i, v):
        v = v + i
        i = i + 1
        return (i, v)

      def my_net(v):
        i = constant_op.constant(0)
        r = control_flow_ops.while_loop(cond,
                                        body, (i, v),
                                        maximum_iterations=10)
        return [r[1]]

      with ops.device('cpu'):
        v = array_ops.placeholder(np.int32, [500])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(my_net, inputs=[v])

      result = sess.run(r, {v: np.zeros([500], np.int32)})
      self.assertAllClose(result[0], np.broadcast_to(45, [500]))

    # Check that there is only one real compile
    self.assert_num_reports(report_helper, 1)
    report = pva.openReport(report_helper.find_report())
    # Check that there is only one execute
    self.assert_number_of_executions(report, 1)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
