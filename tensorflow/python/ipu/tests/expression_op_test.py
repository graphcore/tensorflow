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

import numpy as np

from tensorflow.python import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.ops import array_ops


class CodeletExpressionOpTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testCodeletExpressionOp(self):
    def my_custom_op_1(x, y, z):
      return x * x + y * z

    def my_custom_op_2(x):
      return abs(x)

    def my_net(a, b, c):
      out1 = ipu.custom_ops.codelet_expression_op(my_custom_op_1, a, b, c)
      out2 = ipu.custom_ops.codelet_expression_op(my_custom_op_2, a)
      return [out1, out2]

    with ops.device('cpu'):
      a = array_ops.placeholder(np.float32, [8, 1])
      b = array_ops.placeholder(np.float32, [1])
      c = array_ops.placeholder(np.float32, [1, 8])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
      r = ipu.ipu_compiler.compile(my_net, inputs=[a, b, c])

    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()
    with sl.Session() as sess:
      a_h = np.arange(0, 8).reshape([8, 1])
      b_h = np.arange(1)
      c_h = np.arange(0, 8).reshape([1, 8])

      result = sess.run(r, {a: a_h, b: b_h, c: c_h})
      self.assertAllClose(result[0], my_custom_op_1(a_h, b_h, c_h))
      self.assertAllClose(result[1], my_custom_op_2(a_h))


if __name__ == "__main__":
  googletest.main()
