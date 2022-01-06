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
import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import momentum
from tensorflow.python.platform import googletest
from tensorflow.compat.v1 import disable_v2_behavior

disable_v2_behavior()


class WhileLoopTest(xla_test.XLATestCase):
  def testWhileLoopTupleOfTuples(self):
    with self.session() as sess:
      # This test makes sure that we can handle tuple of tuples for while loops
      random_seed.set_random_seed(1)
      dataType = dtypes.float32
      num_input = 14
      timesteps = 2
      num_units = 128

      def RNN(x):
        # Define a GRU cell with tensorflow
        gru_cell = nn.rnn_cell.GRUCell(num_units, name="GRU")
        # Get gru cell output
        outputs, _ = nn.dynamic_rnn(gru_cell, x, dtype=dataType)
        return outputs[-1]

      def my_net(X, Y):
        # Forward pass
        logits = RNN(X)
        # Loss
        cross_entropy = math_ops.reduce_mean(
            nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=array_ops.stop_gradient(Y)))
        # Training
        train = gradient_descent.GradientDescentOptimizer(0.01).minimize(
            cross_entropy)
        return [cross_entropy, train]

      with ops.device('cpu'):
        X = array_ops.placeholder(dataType, [1, timesteps, num_input])
        Y = array_ops.placeholder(dataType, [1, timesteps, num_units])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(my_net, inputs=[X, Y])

      sess.run(variables.global_variables_initializer())
      result = sess.run(r, {X: np.ones(X.shape), Y: np.ones(Y.shape)})
      # Compare the value - check that the loss is within 1 of the expected
      # value obtained by running on XLA_CPU.
      self.assertAllClose(result[0], 621.9, rtol=1)

  def testGather(self):
    with self.session() as sess:

      def my_net(p, i):
        # Forward pass
        a = array_ops.gather(p, i, axis=0)
        return [a]

      with ops.device('cpu'):
        X = array_ops.placeholder(dtypes.int32, [2, 4])
        Y = array_ops.placeholder(dtypes.int32, [2])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(my_net, inputs=[X, Y])

      sess.run(variables.global_variables_initializer())
      result = sess.run(r, {X: [[1, 3, 5, 7], [0, 2, 4, 6]], Y: [1, 0]})
      self.assertAllClose(result[0], [[0, 2, 4, 6], [1, 3, 5, 7]])

  def testGatherTransposed(self):
    with self.session() as sess:

      def my_net(p, i):
        # Forward pass
        p = array_ops.transpose(p, [1, 0])
        a = array_ops.gather(p, i, axis=0)
        return [a]

      with ops.device('cpu'):
        X = array_ops.placeholder(dtypes.int32, [2, 4])
        Y = array_ops.placeholder(dtypes.int32, [2])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(my_net, inputs=[X, Y])

      sess.run(variables.global_variables_initializer())
      result = sess.run(r, {X: [[1, 3, 5, 7], [0, 2, 4, 6]], Y: [1, 0]})
      self.assertAllClose(result[0], [[3, 2], [1, 0]])

  def testInplaceOpsInRepeats(self):
    with self.session() as sess:

      def my_net(x):
        def cond(i, x):
          del x
          return i < 3

        def body(i, x):
          i = i + 1
          x = nn.relu(x * x)
          return (i, x)

        i = 0
        return control_flow_ops.while_loop(cond, body, (i, x))

      with ops.device('cpu'):
        x = array_ops.placeholder(dtypes.float32, [4])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(my_net, inputs=[x])

      sess.run(variables.global_variables_initializer())
      (c, x) = sess.run(r, {x: np.full([4], 2)})
      self.assertEqual(c, 3)
      self.assertAllClose(x, np.full([4], 256))

  def testNestedWhileLoopsSimplified(self):
    with self.session() as sess:

      def my_net(x):
        def cond(i, x):
          del x
          return i < 3

        def cond1(j, x):
          del x
          return j < 2

        def body1(j, x):
          j = j + 1
          x = x * 2
          return (j, x)

        def body(i, x):
          i = i + 1
          j = 0
          _, x = control_flow_ops.while_loop(cond1,
                                             body1, (j, x),
                                             maximum_iterations=10)
          return (i, x)

        i = 0
        a, b = control_flow_ops.while_loop(cond,
                                           body, (i, x),
                                           maximum_iterations=10)
        return (a, b)

      with ops.device('cpu'):
        x = array_ops.placeholder(dtypes.int32, [4])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(my_net, inputs=[x])

      sess.run(variables.global_variables_initializer())
      c, val = sess.run(r, {x: np.full([4], 2, dtype=np.int32)})
      self.assertEqual(c, 3)
      self.assertAllClose(val, np.full([4], 128))

  def testFusionsInWhileLoops(self):
    with self.session() as sess:

      def my_net():
        def cond(i, x):
          del x
          return i < 3

        def body(i, loss):
          i = i + 1
          init = init_ops.random_normal_initializer(0.0,
                                                    1.0,
                                                    seed=1,
                                                    dtype=np.float32)
          x = variable_scope.get_variable("v2",
                                          dtype=np.float32,
                                          shape=[1, 4, 4, 2],
                                          initializer=init)
          with variable_scope.variable_scope("vs", use_resource=True):
            y = layers.Conv2D(2,
                              1,
                              use_bias=True,
                              kernel_initializer=init_ops.ones_initializer(),
                              name='conv1')(x)
            y = layers.Conv2D(2,
                              1,
                              use_bias=True,
                              kernel_initializer=init_ops.ones_initializer(),
                              name='conv2')(y)
            y = layers.Conv2D(2,
                              1,
                              use_bias=True,
                              kernel_initializer=init_ops.ones_initializer(),
                              name='conv3')(y)
          loss = math_ops.reduce_sum(y)
          optimizer = gradient_descent.GradientDescentOptimizer(0.1)
          train = optimizer.minimize(loss)
          with ops.control_dependencies([train]):
            i = array_ops.identity(i)
            loss = array_ops.identity(loss)
            return (i, loss)

        i = 0
        loss = 0.0
        return control_flow_ops.while_loop(cond,
                                           body, (i, loss),
                                           maximum_iterations=10)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(my_net, inputs=[])

      sess.run(variables.global_variables_initializer())
      c, _ = sess.run(r, {})
      self.assertEqual(c, 3)

  def testTfLstmInWhileV1(self):
    with self.session() as sess:
      dataset = tu.create_dual_increasing_dataset(3,
                                                  data_shape=[4, 1, 8],
                                                  label_shape=[4, 1, 128])

      infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

      def my_net():
        def my_model(loss, x, y):
          with ipu.scopes.ipu_scope("/device:IPU:0"):
            lstm_cell = rnn_cell.LSTMCell(128)
            x, _ = rnn.dynamic_rnn(cell=lstm_cell,
                                   inputs=x,
                                   dtype=dtypes.float32,
                                   time_major=True)

            cross_entropy = nn.softmax_cross_entropy_with_logits_v2(
                logits=x, labels=array_ops.stop_gradient(y))
            loss = math_ops.reduce_mean(cross_entropy)

            optim = gradient_descent.GradientDescentOptimizer(0.01)
            train = optim.minimize(cross_entropy)

            return [loss, train]

        loss = 0.0
        return ipu.loops.repeat(10,
                                my_model, [loss],
                                infeed_queue,
                                use_while_v1=True)

      out = ipu.ipu_compiler.compile(my_net, inputs=[])

      cfg = ipu.config.IPUConfig()
      cfg.ipu_model.compile_ipu_code = False
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      sess.run(infeed_queue.initializer)
      sess.run(variables.global_variables_initializer())
      sess.run(out[0], {})

  def testRepeatLoopGradient(self):
    with self.session() as sess:

      def model(features):
        a = variable_scope.get_variable("a", initializer=1.0)

        def body(x):
          return a * x

        logits = ipu.loops.repeat(5, body, [features])
        loss = math_ops.reduce_sum(logits)
        optimizer = momentum.MomentumOptimizer(learning_rate=.001,
                                               momentum=0.9)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars)
        return a, loss, train_op

      with ops.device('cpu'):
        features = array_ops.placeholder(dtypes.float32, shape=[10])

      with ipu.scopes.ipu_scope('/device:IPU:0'):
        ret = ipu.ipu_compiler.compile(model, [features])

      cfg = ipu.config.IPUConfig()
      cfg.ipu_model.compile_ipu_code = False
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      sess.run(variables.global_variables_initializer())
      x, _ = sess.run(ret, feed_dict={features: np.ones([10])})
      self.assertEqual(x, 1)

  def testWhileLoopAliasing1(self):
    # Checks that the 'add' isn't in-place, that the 10 operand
    # of the add can also be fed to the second output and the
    # first output can be fed from the add.  Output 0 is not
    # an alias of anything, but output 1 is an alias of input 0
    with self.session() as sess:

      def body(a, b):
        c = a + b
        return [c, a]

      def my_net(a, b):
        r = ipu.loops.repeat(4, body, [a, b])
        return r

      with ops.device('cpu'):
        a = array_ops.placeholder(np.float32, [])
        b = array_ops.placeholder(np.float32, [])

      with ops.device("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(my_net, inputs=[a, b])

      fd = {
          a: 1.0,
          b: 0.0,
      }
      result = sess.run(res, fd)

      # While loop is generating the Fibonacci sequence
      self.assertAllClose(result[0], 5.0)
      self.assertAllClose(result[1], 3.0)

  def testWhileLoopAliasing2(self):
    # Checks that the 'add' is not in-place. Output 0 is an identical
    # copy of input 1, and output 1 is not an alias of anything.
    # Output 1 is the result of the add.
    with self.session() as sess:

      def body(a, b):
        c = a + b
        return [a, c]

      def my_net(a, b):
        r = ipu.loops.repeat(4, body, [a, b])
        return r

      with ops.device('cpu'):
        a = array_ops.placeholder(np.float32, [])
        b = array_ops.placeholder(np.float32, [])

      with ops.device("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(my_net, inputs=[a, b])

      fd = {
          a: 1.0,
          b: 0.0,
      }
      result = sess.run(res, fd)

      self.assertAllClose(result[0], 1.0)
      self.assertAllClose(result[1], 4.0)

  # This test is to ensure that the issue exposed by T10605 is fixed.
  def testWhileLoopInplaceAlias(self):
    def my_net():

      coordinates = constant_op.constant(0.1, shape=[5], dtype=dtypes.float32)
      ta = tensor_array_ops.TensorArray(coordinates.dtype,
                                        coordinates.get_shape()[0],
                                        element_shape=[]).unstack(coordinates)

      def loop_body(i, summation):
        factor = 0.5
        summand = ta.read(i) * factor * constant_op.constant(1., shape=[2])
        # if we don't use the TensorArray but just hardcode the constant 0.01 the problem goes away
        return math_ops.add(i, 1), math_ops.add(summation, summand)

      return control_flow_ops.while_loop(
          cond=lambda i, _: i < 5,
          body=loop_body,
          loop_vars=(constant_op.constant(0, dtype=dtypes.int32),
                     constant_op.constant(0.0, shape=[2],
                                          dtype=dtypes.float32)))[1]

    # Checks that the 'add' is not in-place. Output 0 is an identical
    # copy of input 1, and output 1 is not an alias of anything.
    # Output 1 is the result of the add.
    with self.session() as sess:

      with ops.device('cpu'):
        array_ops.placeholder(np.float32, [])

      with ops.device("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(my_net)

      result = sess.run(res)
      print(result)
      self.assertAllClose(result[0], [0.25, 0.25])


if __name__ == "__main__":
  googletest.main()
