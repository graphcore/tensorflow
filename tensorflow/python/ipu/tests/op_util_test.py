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
from tensorflow.python.ipu import test_utils as tu
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python import ipu


class OpUtilTest(test_util.TensorFlowTestCase):
  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testAccumulateGradients(self):
    """
    Check that accumulate_gradients creates the correct type buffers when the
    weights are alternating in dtype.
    """
    x = constant_op.constant(1.0, shape=[3, 3])
    for i in range(4):
      w = variable_scope.get_variable(name=f"matmul{i}/w",
                                      shape=[3, 3],
                                      dtype=dtypes.float16 if i %
                                      2 == 0 else dtypes.float32,
                                      initializer=init_ops.ones_initializer())
      x = math_ops.matmul(math_ops.cast(x, w.dtype), w)
    opt = gradient_descent.GradientDescentOptimizer(0.1)
    grads_and_vars = opt.compute_gradients(x)

    acc = ipu.ops.op_util.accumulate_gradients(grads_and_vars, None)
    for acc_grad, var in acc:
      # Use base_dtype to ignore reference dtypes.
      self.assertEqual(acc_grad.dtype.base_dtype, var.dtype.base_dtype)


if __name__ == "__main__":
  googletest.main()
