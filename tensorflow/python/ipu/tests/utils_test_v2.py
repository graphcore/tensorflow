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
"""
This test suite is analogous in purpose to test_utils.py.
However, in test_utils.py, disable_v2_behaviour is called.
"""

import numpy as np

from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.eager import def_function
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class ContribIpuOpsTestWithV2Behaviour(test_util.TensorFlowTestCase):
  def testRepeatLoopWithConditional(self):
    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      opt = keras.optimizer_v2.gradient_descent.SGD(0.01)

      # Generate some dummy data.
      a_data = np.ones((1, 10), dtype=np.float32)

      v = variables.Variable(1, shape=(), dtype=np.float32)

      @def_function.function(experimental_compile=True)
      def fn_with_loop(in_a, n):
        def loop_body(a, c):
          a, c = control_flow_ops.cond(math_ops.less_equal(c, n), lambda:
                                       (a + v, c), lambda: (a, c))
          return a, c + 1

        # Run the loop max 10 runs.
        with GradientTape() as tape:
          tape.watch(v)
          res = ipu.loops.repeat(10, loop_body, inputs=[in_a, 0])
          cost = math_ops.reduce_sum(res[0])
          g = tape.gradient(cost, v)
          opt.apply_gradients([(g, v)])
        return cost

      last_loss = float('inf')
      for _ in range(5):
        c = np.array([0], dtype=np.int32)
        res = strategy.run(fn_with_loop, args=[a_data, c])
        loss_sum = np.sum(res)
        self.assertLess(loss_sum, last_loss)
        last_loss = loss_sum


if __name__ == "__main__":
  googletest.main()
