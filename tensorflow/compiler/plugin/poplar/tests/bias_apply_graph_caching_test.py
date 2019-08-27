#  Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


class BiasApplyGraphCachingTest(xla_test.XLATestCase):
  def testMatch(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        biases1 = array_ops.placeholder(np.float32, shape=[2])
        biases2 = array_ops.placeholder(np.float32, shape=[2])
        biases3 = array_ops.placeholder(np.float32, shape=[2])
        grads1 = array_ops.placeholder(np.float32, shape=[2, 10])
        grads2 = array_ops.placeholder(np.float32, shape=[2, 10])
        grads3 = array_ops.placeholder(np.float32, shape=[2, 10])
        vlr = array_ops.placeholder(np.float32, shape=[])

        def bias_apply(bias, grad, lr):
          return bias - math_ops.reduce_sum(grad, axis=1) * lr

        out = (bias_apply(biases1, grads1, vlr) + bias_apply(
            biases2, grads2, 0.1) + bias_apply(biases3, grads3, 0.2))

        with ops.device('cpu'):
          report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(True, True, True)

      sess.run(variables.global_variables_initializer())

      sess.run(report)

      r = sess.run(
          out, {
              biases1: np.ones([2]),
              biases2: np.ones([2]),
              biases3: np.ones([2]),
              grads1: np.ones([2, 10]),
              grads2: np.ones([2, 10]),
              grads3: np.ones([2, 10]),
              vlr: 0.1
          })
      self.assertAllClose(r, [-1., -1.])
      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      self.assertEqual(
          tu.count_compute_sets_matching(cs_list, '*ReduceOnTile*'), 1)

  def testMatchBecauseEvenWhenNotInplace(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        biases1 = array_ops.placeholder(np.float32, shape=[2])
        grads1 = array_ops.placeholder(np.float32, shape=[2, 10])
        grads2 = array_ops.placeholder(np.float32, shape=[2, 10])

        def bias_apply(bias, grad):
          return bias - math_ops.reduce_sum(grad, axis=1) * 0.1

        out = bias_apply(biases1, grads1) + bias_apply(biases1, grads2)

        with ops.device('cpu'):
          report = gen_ipu_ops.ipu_event_trace()

      tu.configure_ipu_system(True, True, True)

      sess.run(variables.global_variables_initializer())

      sess.run(report)

      r = sess.run(
          out, {
              biases1: np.ones([2]),
              grads1: np.ones([2, 10]),
              grads2: np.ones([2, 10])
          })
      self.assertAllClose(r, [0., 0.])
      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)
      # We still reuse the code even though only one reduce is inplace.
      self.assertEqual(
          tu.count_compute_sets_matching(cs_list, '*ReduceOnTile*'), 1)


if __name__ == "__main__":
  googletest.main()
