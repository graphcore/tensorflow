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
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.plugin.poplar.tests.test_utils import ReportJSON
from tensorflow.python import ipu
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


def count_event_type(events, event_type):
  return sum([1 if x.type == event_type else 0 for x in events])


class WhileLoopPerfTest(xla_test.XLATestCase):
  def testIpuWhilePerfTest(self):
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

      report = ReportJSON(self, sess)
      report.reset()

      result = sess.run(r, {v: np.zeros([500], np.int32)})
      self.assertAllClose(result[0], np.broadcast_to(45, [500]))

      report.parse_log()

      # Check that there is only one real compile
      report.assert_contains_one_compile_event()

      # Check that there is only one execute
      report.assert_num_execution_reports_equal(1)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
