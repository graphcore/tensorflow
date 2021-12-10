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

import numpy as np
from absl.testing import parameterized

from tensorflow.python.framework import test_util
from tensorflow.python import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


class ControlFlowOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  @parameterized.parameters((True,), (False,))
  @test_util.deprecated_graph_mode_only
  def test_barrier(self, gradients_barrier):
    a_val = array_ops.ones([2, 2], dtype=np.float32)
    b_val = array_ops.ones([2, 8], dtype=np.float32)

    def model_fn():
      a_var = variable_scope.get_variable("a", initializer=a_val)
      b_var = variable_scope.get_variable("b", initializer=b_val)
      a = a_var * 4
      b = b_var * 2
      # Adding the barrier here forces a_var * 4 and b_var * 2 to execute before
      # a_var * 8.
      a, b = ipu.control_flow_ops.barrier(
          [a, b], insert_barrier_for_gradients=gradients_barrier)
      a = a + a_var * 8
      # Not a real loss function, but good enough for testing backprop.
      loss = math_ops.reduce_sum(a @ b)
      outputs = gradients_impl.gradients(loss, [a_var, b_var])
      outputs.append(loss)
      return outputs

    with sl.Session() as sess:
      with ipu.scopes.ipu_scope('/device:IPU:0'):
        with variable_scope.variable_scope("vs", use_resource=True):
          output = ipu.ipu_compiler.compile(model_fn, [])
      ipu.utils.move_variable_initialization_to_cpu()
      sess.run(variables.global_variables_initializer())
      x = sess.run(output)

    self.assertAllClose(x[0], np.full([2, 2], 192))
    self.assertAllClose(x[1], np.full([2, 8], 48))
    self.assertAllClose(x[2], 768.0)


if __name__ == "__main__":
  test.main()
