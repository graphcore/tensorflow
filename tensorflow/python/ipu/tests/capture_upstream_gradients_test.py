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

from tensorflow.python.eager import def_function
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import config
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu.eager.backprop import GradientCaptureContext
from tensorflow.python.ipu.eager.backprop import GradientCaptureTape
from tensorflow.python.ipu.ops import functional_ops
from tensorflow.python.ipu.ops.grad_util_ops import capture_upstream_gradients
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class CaptureUpstreamGradientsTest(test_util.TensorFlowTestCase):
  def testWrongGradientTape(self):
    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():

      @def_function.function(jit_compile=True)
      def f(x):
        with GradientTape() as _:
          y = capture_upstream_gradients(x, tag="tanh_grad")
          z = math_ops.tanh(y)
        return z

    with self.assertRaisesRegex(RuntimeError,
                                "may only be used within the context of "):
      x = np.ones(4, dtype=np.float16)
      _ = strategy.run(f, [x])

  def testForwardPassTape(self):
    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():

      @def_function.function(jit_compile=True)
      def f(x):
        with GradientCaptureTape() as _:
          y = math_ops.tanh(capture_upstream_gradients(x, tag="tanh_grad"))
          z = math_ops.tanh(x)
        return y, z

      x = np.ones(4, dtype=np.float16)
      y, z = strategy.run(f, [x])
      self.assertAllEqual(np.squeeze(y), z)

  def testForwardPassContext(self):
    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():

      @def_function.function(jit_compile=True)
      def f(x):
        with GradientCaptureContext() as _:
          y = math_ops.tanh(capture_upstream_gradients(x, tag="tanh_grad"))
          z = math_ops.tanh(x)
        return y, z

      x = np.ones(4, dtype=np.float16)
      y, z = strategy.run(f, [x])
      self.assertAllEqual(np.squeeze(y), z)

  def testVariableAndActivationTape(self):
    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      x = variables.Variable(3.0)

      # Verify the captured grads.
      @def_function.function(jit_compile=True)
      def f():
        with GradientCaptureTape() as tape:
          o = x**2
          p = capture_upstream_gradients(o, tag="tanh_grad")
          y = math_ops.tanh(p)

        dfdx = tape.gradient(y, x)
        dfda = tape.captured_gradients
        dfda_manual = gen_math_ops.tanh_grad(y, array_ops.ones_like(y))

        return dfdx, dfda, dfda_manual

      dfdx, dfda, dfda_manual = strategy.run(f)
      self.assertAllEqual(dfda['tanh_grad'], dfda_manual)

      # Now verify the grads w.r.t variables match.
      @def_function.function(jit_compile=True)
      def g():
        with GradientTape() as tape:
          o = x**2
          y = math_ops.tanh(o)

        return tape.gradient(y, x)

      dfdx_vanilla_grad_tape = strategy.run(g)
      self.assertAllEqual(dfdx, dfdx_vanilla_grad_tape)

  def testVariableAndActivationContext(self):
    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      opt = gradient_descent.SGD()
      x = variables.Variable(3.0)

      # Verify the captured grads.
      @def_function.function(jit_compile=True)
      def f():
        with GradientCaptureContext() as gcc:
          o = x**2
          p = capture_upstream_gradients(o, tag="tanh_grad")
          y = math_ops.tanh(p)
          dfdx = opt.get_gradients(y, x)

        dfda = gcc.captured_gradients
        dfda_manual = gen_math_ops.tanh_grad(y, array_ops.ones_like(y))

        return dfdx, dfda, dfda_manual

      dfdx, dfda, dfda_manual = strategy.run(f)
      self.assertAllEqual(dfda['tanh_grad'], dfda_manual)

      # Now verify the grads w.r.t variables match.
      @def_function.function(jit_compile=True)
      def g():
        o = x**2
        y = math_ops.tanh(o)
        return opt.get_gradients(y, x)

      dfdx_vanilla_grad_tape = strategy.run(g)
      self.assertAllEqual(dfdx, dfdx_vanilla_grad_tape)

  def testVariableAndActivationOutlinedFunctionContext(self):
    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      opt = gradient_descent.SGD()
      x = variables.Variable(3.0)

      @functional_ops.outlined_function
      def h(v):
        o = v**2
        p = capture_upstream_gradients(o, tag="tanh_grad")
        return math_ops.tanh(p)

      # Verify the captured grads.
      @def_function.function(jit_compile=True)
      def f():
        with GradientCaptureContext() as gcc:
          y = h(x)
          dfdx = opt.get_gradients(y, x)

        dfda = gcc.captured_gradients
        dfda_manual = gen_math_ops.tanh_grad(y, array_ops.ones_like(y))

        return dfdx, dfda, dfda_manual

      dfdx, dfda, dfda_manual = strategy.run(f)
      self.assertAllEqual(dfda['tanh_grad'], dfda_manual)

      # Now verify the grads w.r.t variables match.
      @def_function.function(jit_compile=True)
      def g():
        o = x**2
        y = math_ops.tanh(o)
        return opt.get_gradients(y, x)

      dfdx_vanilla_grad_tape = strategy.run(g)
      self.assertAllEqual(dfdx, dfdx_vanilla_grad_tape)

  def testVariableAndActivationOutlinedFunctionTape(self):
    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      opt = gradient_descent.SGD()
      x = variables.Variable(3.0)

      @functional_ops.outlined_function
      def h(v):
        o = v**2
        p = capture_upstream_gradients(o, tag="tanh_grad")
        return math_ops.tanh(p)

      # Verify the captured grads.
      @def_function.function(jit_compile=True)
      def f():
        with GradientCaptureTape() as tape:
          y = h(x)
          dfdx = opt.get_gradients(y, x)

        dfda = tape.captured_gradients
        dfda_manual = gen_math_ops.tanh_grad(y, array_ops.ones_like(y))

        return dfdx, dfda, dfda_manual

      dfdx, dfda, dfda_manual = strategy.run(f)
      self.assertAllEqual(dfda['tanh_grad'], dfda_manual)

      # Now verify the grads w.r.t variables match.
      @def_function.function(jit_compile=True)
      def g():
        o = x**2
        y = math_ops.tanh(o)
        return opt.get_gradients(y, x)

      dfdx_vanilla_grad_tape = strategy.run(g)
      self.assertAllEqual(dfdx, dfdx_vanilla_grad_tape)


if __name__ == "__main__":
  googletest.main()
