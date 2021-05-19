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

import os
import numpy as np

from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.compiler.tests import xla_test
from tensorflow.python.compiler.xla import xla
from tensorflow.python.platform import googletest
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.config import IPUConfig


class StatefulGradientAccumulateTest(xla_test.XLATestCase):
  def testStatefulGradientAccumulate(self):
    with self.session() as sess:
      dtype = np.float32

      def my_net(y):
        def cond(i, x, y):
          del x
          del y
          return i < 10

        def body(i, x, y):
          x = x + gen_poputil_ops.ipu_stateful_gradient_accumulate(
              array_ops.ones_like(x), num_mini_batches=5, verify_usage=False)
          y = y + array_ops.ones_like(x)
          i = i + 1
          return (i, x, y)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, y, y))

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])

      opts = IPUConfig()
      opts.configure_ipu_system()

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      y = sess.run(r, {y: [10]})
      self.assertEqual(y[0], 10)
      self.assertAllEqual(y[1], [20])
      self.assertAllEqual(y[2], [20])

  def testStatefulGradientAccumulateInvalidUse(self):
    with self.session() as sess:
      dtype = np.float32

      def my_net(y):
        def cond(i, x, y):
          del x
          del y
          return i < 10

        def body(i, x, y):
          x = x + gen_poputil_ops.ipu_stateful_gradient_accumulate(
              array_ops.ones_like(x), num_mini_batches=5)
          y = y + array_ops.ones_like(x)
          i = i + 1
          return (i, x, y)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, y, y))

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])

      opts = IPUConfig()
      opts.configure_ipu_system()

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      with self.assertRaisesRegex(errors.FailedPreconditionError,
                                  "The .*IpuStatefulGradientAccumulate op"):
        sess.run(r, {y: [10]})

  def testLoopRepeatCountDoesntDivide(self):
    with self.session() as sess:
      dtype = np.float32

      def my_net(y):
        def cond(i, x, y):
          del x
          del y
          return i < 10

        def body(i, x, y):
          x = x + gen_poputil_ops.ipu_stateful_gradient_accumulate(
              array_ops.ones_like(x), num_mini_batches=4, verify_usage=False)
          y = y + array_ops.ones_like(x)
          i = i + 1
          return (i, x, y)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, y, y))

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])

      opts = IPUConfig()
      opts.configure_ipu_system()

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          "Detected a gradient accumulation operation with 4 number of mini "
          "batches inside a loop with 10 iterations."):
        sess.run(r, {y: [10]})

  def testStatefulGradientAccumulateWithMomentum(self):
    with self.session() as sess:
      dtype = np.float32

      def get_var():
        with variable_scope.variable_scope("",
                                           use_resource=True,
                                           reuse=variable_scope.AUTO_REUSE):
          var = variable_scope.get_variable(
              "x",
              dtype=dtype,
              shape=[1],
              initializer=init_ops.constant_initializer(0.0))
        return var

      def my_net(y):
        def cond(i, x):
          del x
          return i < 10

        def body(i, x):
          var = get_var()
          x = x + \
              gen_poputil_ops.ipu_stateful_gradient_accumulate_with_momentum(
                  var.handle,
                  array_ops.ones_like(x),
                  momentum=0.8,
                  num_mini_batches=5)
          i = i + 1
          return (i, x)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, y))

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])

      opts = IPUConfig()
      opts.configure_ipu_system()

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      sess.run(variables.global_variables_initializer())
      y = sess.run(r, {y: [10]})
      self.assertEqual(y[0], 10)
      self.assertAllClose(y[1], [24.0])
      var = sess.run(get_var())
      self.assertAllClose(var, [9.0])

  def testStatefulGradientAccumulateWithMomentumInvalid(self):
    with self.session() as sess:
      dtype = np.float32

      def get_var():
        with variable_scope.variable_scope("",
                                           use_resource=True,
                                           reuse=variable_scope.AUTO_REUSE):
          var = variable_scope.get_variable(
              "x",
              dtype=dtype,
              shape=[1],
              initializer=init_ops.constant_initializer(0.0))
        return var

      def my_net(y):
        def cond(i, x):
          del x
          return i < 8

        def body(i, x):
          var = get_var()
          x = x + \
              gen_poputil_ops.ipu_stateful_gradient_accumulate_with_momentum(
                  var.handle,
                  array_ops.ones_like(x),
                  momentum=0.8,
                  num_mini_batches=5)
          i = i + 1
          return (i, x)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, y))

      with ops.device('cpu'):
        y = array_ops.placeholder(dtype, [1])

      opts = IPUConfig()
      opts.configure_ipu_system()

      with ops.device("/device:IPU:0"):
        r = xla.compile(my_net, inputs=[y])

      sess.run(variables.global_variables_initializer())
      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          "Detected a gradient accumulation operation with 5 number of mini "
          "batches inside a loop with 8 iterations."):
        sess.run(r, {y: [10]})


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
