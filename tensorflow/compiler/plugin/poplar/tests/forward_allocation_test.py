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
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ipu import utils
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables


class ForwardAllocationTest(xla_test.XLATestCase):
  def testPrefixPathWithReshape(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
        z = array_ops.placeholder(np.float32, shape=[32])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            use_bias=True,
                            kernel_initializer=init_ops.ones_initializer())(x)
        res = gen_array_ops.reshape(y, [32]) + z

      opts = utils.create_ipu_config()
      utils.configure_ipu_system(opts)

      sess.run(variables.global_variables_initializer())

      result = sess.run(res, {
          x: np.reshape(np.arange(32), [1, 4, 4, 2]),
          z: np.ones([32])
      })
      # Confirmed with values on the CPU.
      self.assertAllClose(result, [
          2., 2., 6., 6., 10., 10., 14., 14., 18., 18., 22., 22., 26., 26.,
          30., 30., 34., 34., 38., 38., 42., 42., 46., 46., 50., 50., 54., 54.,
          58., 58., 62., 62.
      ])

  def testPrefixPathWithTranspose(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
        z = array_ops.placeholder(np.float32, shape=[4, 4, 2, 1])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            use_bias=True,
                            kernel_initializer=init_ops.ones_initializer())(x)
        res = array_ops.transpose(y, [1, 2, 3, 0]) + z

      opts = utils.create_ipu_config()
      utils.configure_ipu_system(opts)

      sess.run(variables.global_variables_initializer())

      result = sess.run(res, {
          x: np.reshape(np.arange(32), [1, 4, 4, 2]),
          z: np.ones([4, 4, 2, 1])
      })
      self.assertAllClose(
          result,
          [[[[2.], [2.]], [[6.], [6.]], [[10.], [10.]], [[14.], [14.]]],
           [[[18.], [18.]], [[22.], [22.]], [[26.], [26.]], [[30.], [30.]]],
           [[[34.], [34.]], [[38.], [38.]], [[42.], [42.]], [[46.], [46.]]],
           [[[50.], [50.]], [[54.], [54.]], [[58.], [58.]], [[62.], [62.]]]])

  def testPrefixPathWithElementwiseInPath(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
        z = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
        s = array_ops.placeholder(np.float32, shape=[])

        with variable_scope.variable_scope("vs", use_resource=True):
          y = layers.Conv2D(2,
                            1,
                            use_bias=True,
                            kernel_initializer=init_ops.ones_initializer())(x)
        res = y + z * s

      opts = utils.create_ipu_config()
      utils.configure_ipu_system(opts)

      sess.run(variables.global_variables_initializer())

      result = sess.run(
          res, {
              x: np.reshape(np.arange(32), [1, 4, 4, 2]),
              z: np.reshape(np.arange(32), [1, 4, 4, 2]),
              s: 2.0
          })
      # Confirmed with values on the CPU.
      self.assertAllClose(
          result, [[[[1., 3.], [9., 11.], [17., 19.], [25., 27.]],
                    [[33., 35.], [41., 43.], [49., 51.], [57., 59.]],
                    [[65., 67.], [73., 75.], [81., 83.], [89., 91.]],
                    [[97., 99.], [105., 107.], [113., 115.], [121., 123.]]]])

  def testPrefixPathWithCast(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        data = array_ops.placeholder(np.float32, shape=[1, 7, 1])
        kernel2 = array_ops.placeholder(np.float16, shape=[3, 1, 1])
        kernel = math_ops.cast(kernel2, np.float32)

        with variable_scope.variable_scope("vs", use_resource=True):
          res = nn.conv1d(data, kernel, stride=1, padding="VALID")

      opts = utils.create_ipu_config()
      utils.configure_ipu_system(opts)

      sess.run(variables.global_variables_initializer())

      result = sess.run(
          res, {
              data: np.reshape(np.array([1, 0, 2, 3, 0, 1, 1]), [1, 7, 1]),
              kernel2: np.reshape(np.array([2, 1, 3]), [3, 1, 1])
          })
      # Confirmed with values on the CPU.
      self.assertAllClose(result,
                          [np.reshape(np.array([8, 11, 7, 9, 4]), [5, 1])])


if __name__ == "__main__":
  googletest.main()
