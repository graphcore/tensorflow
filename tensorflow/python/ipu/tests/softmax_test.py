# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
from absl.testing import parameterized

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python import ipu
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest
from tensorflow.python.ipu.ops import nn_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops as tf_nn_ops


class PoplibsSoftmaxTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  configured = False

  def __configureIPU(self):
    if not self.configured:
      cfg = ipu.config.IPUConfig()
      cfg.ipu_model.compile_ipu_code = False
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()
      self.configured = True

  def _softmax_cpu(self, x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex, axis=0)

  @parameterized.named_parameters(("Stable", 100, True),
                                  ("Unstable", 100, False))
  @test_util.deprecated_graph_mode_only
  def testSoftmax(self, n, stable):
    with self.session() as sess:
      self.__configureIPU()

      i_h = np.linspace(-10, 10, n, dtype='float32')
      ref_h = self._softmax_cpu(i_h)

      with ops.device("/device:IPU:0"):
        i = array_ops.placeholder(np.float32, shape=[n])
        o = nn_ops.softmax(i, stable=stable)

        test_h = sess.run(o, {i: i_h})

        self.assertAllClose(ref_h, test_h)

  @parameterized.named_parameters(("Stable", 100, True),
                                  ("Unstable", 100, False))
  @test_util.deprecated_graph_mode_only
  def testSoftmaxGrad(self, n, stable):
    with self.session() as sess:
      self.__configureIPU()

      x_h = np.linspace(-10, 10, n, dtype='float32')

      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[n])
        y = nn_ops.softmax(x, stable=stable)

        # Directly compute the gradients
        x_grad = gradients_impl.gradients(y, x)

        # Use the tf.nn.softmax for comparison
        y_ref = tf_nn_ops.softmax(x)
        x_grad_ref = gradients_impl.gradients(y_ref, x)

        (test_h, ref_h) = sess.run((x_grad, x_grad_ref), {x: x_h})

        self.assertAllClose(test_h, ref_h)


if __name__ == "__main__":
  googletest.main()
