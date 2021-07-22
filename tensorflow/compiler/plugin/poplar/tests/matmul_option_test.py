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

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent
from tensorflow.python.ipu.config import IPUConfig


class IpuXlaMatMulOptionTest(xla_test.XLATestCase):
  def testMatMulFwdBackwd(self):
    with self.session() as sess:

      cfg = IPUConfig()
      cfg.ipu_model.compile_ipu_code = False
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("vs", use_resource=True):
          w1 = variable_scope.get_variable(
              "w1",
              shape=[4, 3],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(
                  np.array([[1, 2, 1], [1, 3, 4], [1, 5, 6], [1, 7, 8]],
                           dtype=np.float32)))
          b1 = variable_scope.get_variable(
              "b1",
              shape=[3],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(
                  np.array([2, 1, 1], dtype=np.float32)))
          w2 = variable_scope.get_variable(
              "w2",
              shape=[3, 2],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(
                  np.array([[3, 4], [5, 6], [7, 8]], dtype=np.float32)))
          b2 = variable_scope.get_variable(
              "b2",
              shape=[2],
              dtype=np.float32,
              initializer=init_ops.constant_initializer(
                  np.array([2, 1], dtype=np.float32)))

        x = array_ops.placeholder(np.float32, shape=[3, 4])
        y = math_ops.matmul(x, w1) + b1
        y = math_ops.matmul(y, w2) + b2

        expected = array_ops.placeholder(np.float32, shape=[3, 2])
        xent = nn.softmax_cross_entropy_with_logits_v2(
            logits=y, labels=array_ops.stop_gradient(expected))

        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(xent)

        fd = {
            x:
            np.array([[7, 3, 5, 9], [1, 2, 3, 4], [5, 6, 7, 8]],
                     dtype=np.float32),
            expected: [[1, 2], [3, 4], [5, 6]]
        }

        sess.run(variables.global_variables_initializer())
        sess.run(train, feed_dict=fd)


if __name__ == "__main__":
  googletest.main()
