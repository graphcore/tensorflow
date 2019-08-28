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
import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


class MatMulGraphCachingTest(xla_test.XLATestCase):
  def testMatMulBroadcast(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        in0 = array_ops.placeholder(np.float16, shape=[1024])
        in0_bcast = gen_array_ops.broadcast_to(in0, shape=[1024, 1024])
        in1 = array_ops.placeholder(np.float16, shape=[1024, 1024])

        with variable_scope.variable_scope("vs", use_resource=True):
          weights = variable_scope.get_variable(
              "x",
              dtype=np.float16,
              shape=[1024, 1024],
              initializer=init_ops.constant_initializer(0.0))

        mm1 = math_ops.matmul(in0_bcast, weights, name="mm1")
        mm2 = math_ops.matmul(in1, mm1, name="mm2")
        with ops.device('cpu'):
          report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(True, True, True)
      tu.move_variable_initialization_to_cpu()

      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run(mm2, {in0: np.zeros(in0.shape), in1: np.zeros(in1.shape)})

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)

      max_tile_size = tu.get_maximum_tile_size_from_events(s)
      self.assertAllInRange([max_tile_size], 140000, 160000)

      cs_list = tu.get_compute_sets_from_report(s)
      ok = ['__seed*', 'host-exchange-local-copy-', 'mm1/dot*', 'Copy_']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))


if __name__ == "__main__":
  googletest.main()
