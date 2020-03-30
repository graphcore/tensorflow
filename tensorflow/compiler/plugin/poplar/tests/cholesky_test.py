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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.training import gradient_descent


class IpuXlaCholeskyTest(xla_test.XLATestCase):
  # Overriding abstract method.
  def cached_session(self):
    return 0

  # Overriding abstract method.
  def test_session(self):
    return 0

  def testCholeskyFwdBackwd(self):
    with self.session() as sess:

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):
          ls = variable_scope.get_variable(
              "lengthscale",
              shape=[1],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(
                  np.array([0.2], dtype=np.float32)))

        x = array_ops.placeholder(np.float32, shape=[4, 2])
        expected = array_ops.placeholder(np.float32, shape=[4, 1])

        x_t = array_ops.transpose(array_ops.expand_dims(x, 0), (1, 0, 2))
        K = math_ops.exp(-0.5 * math_ops.reduce_sum(((x_t - x) / ls)**2., -1))
        L = linalg_ops.cholesky(K)
        alpha = linalg_ops.cholesky_solve(L, expected)

        loss = -math_ops.reduce_mean(
            math_ops.matmul(alpha, expected, transpose_a=True))
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)

        fd = {
            x:
            np.array([[1., 0.63920265], [0.63920265, 1.],
                      [0.30846608, 0.24088137], [0.38437635, 0.76085484]],
                     dtype=np.float32),
            expected: [[0.4662998], [-0.27042738], [-0.1996377], [-1.1648941]]
        }

        sess.run(variables.global_variables_initializer())
        sess.run(train, feed_dict=fd)


if __name__ == "__main__":
  googletest.main()
