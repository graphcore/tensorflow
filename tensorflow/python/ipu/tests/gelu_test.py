# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.client import session as se
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python import ipu
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent


class PopnnGeluTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testGelu(self):
    def test_approx_gelu(x):
      return 0.5 * x * (
          1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def ipu_gelu(x):
      x = ipu.ops.nn_ops.gelu(x)
      return [x]

    for test_type in [[np.float16, 1e-2], [np.float32, 1e-7]]:
      with ops.device('cpu'):
        input_data = array_ops.placeholder(test_type[0], shape=[10, 20])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(ipu_gelu, inputs=[input_data])

      with se.Session() as sess:
        in_data = np.random.rand(10, 20)
        ipu_result = sess.run(r, {input_data: in_data})

      self.assertAllClose(ipu_result, [test_approx_gelu(np.array(in_data))],
                          rtol=test_type[1])

  @test_util.deprecated_graph_mode_only
  def testGeluGrad(self):
    a_size = 5
    b_size = 6
    ab_size = 10
    mat_values = [
        -5.0, -1.2, -1.0, -0.5, -0.2, -0.15, -0.1, 0.0, 0.1, 0.15, 0.2, 0.5,
        1.0, 1.2, 5.0
    ]

    def test_sech(x):
      return 2.0 / (np.exp(x) + np.exp(-x))

    def test_approx_derivative_gelu(x):
      return 0.5 * np.tanh(0.0356774*np.power(x, 2) + 0.797885*x) + \
      (0.0535161*np.power(x, 3) + 0.398942*x)* \
      np.power(test_sech(0.0356774*np.power(x, 3) + 0.797885*x), 2) + 0.5

    def ipu_gelu_back(a, b):
      w = math_ops.matmul(a, b)
      gelu_output = ipu.ops.nn_ops.gelu(w)
      cost = gelu_output
      opt = gradient_descent.GradientDescentOptimizer(learning_rate=0.1)
      gradients = opt.compute_gradients(cost, w)

      return [gelu_output, gradients, w]

    for mat_value in mat_values:
      for test_type in [[np.float16, 1e-2], [np.float32, 1e-2]]:
        with ops.device('cpu'):
          a = array_ops.placeholder(test_type[0], shape=[a_size, ab_size])
          b = array_ops.placeholder(test_type[0], shape=[ab_size, b_size])

        with ipu.scopes.ipu_scope("/device:IPU:0"):
          r = ipu.ipu_compiler.compile(ipu_gelu_back, inputs=[a, b])

        with se.Session() as sess:
          in_a = np.full((a_size, ab_size), mat_value)
          in_b = np.full((ab_size, b_size), mat_value)
          res = sess.run(r, {a: in_a, b: in_b})

          gradients_res_values = res[1][0][1]
          gradients_res_grads = res[1][0][0]
          variable_values = res[2]

          self.assertAllClose(variable_values,
                              gradients_res_values,
                              rtol=test_type[1])
          self.assertEqual(gradients_res_grads.shape, (a_size, b_size))
          self.assertAllClose(test_approx_derivative_gelu(mat_value *
                                                          mat_value * ab_size),
                              gradients_res_grads[0][0],
                              rtol=test_type[1])


if __name__ == "__main__":
  googletest.main()
