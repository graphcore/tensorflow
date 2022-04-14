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

import numpy as np

from tensorflow.python import ipu
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops as nn
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import gradient_descent as gd
from tensorflow.python.ipu.optimizers import map_gradient_optimizer

WEIGHT_DECAY = 0.01


def map_fn_quadratic(grad, var):
  return math_ops.square(grad)


def map_fn_clipping_7_and_14(grad, var):
  return clip_ops.clip_by_value(grad, 7, 14, name=None)


def map_fn_decay(grad, var):
  return grad + (WEIGHT_DECAY * var)


class MapGradientOptimizerTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testMapGradientOptimizer(self):
    # test with map_fn_quadratic(), x + y + z
    with self.cached_session():
      optimizer = gd.GradientDescentOptimizer(3.0)
      values = [1.0, 2.0, 3.0]
      vars_ = [variables.Variable([v], dtype=dtypes.float32) for v in values]
      map_optimizer = map_gradient_optimizer.MapGradientOptimizer(
          optimizer, map_fn_quadratic)
      grads_and_vars = map_optimizer.compute_gradients(
          vars_[0] + vars_[1] + vars_[2], vars_)

      variables.global_variables_initializer().run()
      expect_grads = ([1], [1], [1])
      index = 0
      for grad, _ in grads_and_vars:
        self.assertAllCloseAccordingToType(expect_grads[index],
                                           self.evaluate(grad))
        index += 1

  @test_util.deprecated_graph_mode_only
  def testMapGradientDecentWithSquare(self):
    # test with map_fn_quadratic(), x^2 + y + z
    with self.cached_session():
      optimizer = gd.GradientDescentOptimizer(3.0)
      values = [1.0, 1.0, 1.0]
      vars_ = [variables.Variable([v], dtype=dtypes.float32) for v in values]
      map_optimizer = map_gradient_optimizer.MapGradientOptimizer(
          optimizer, map_fn_quadratic)
      grads_and_vars = map_optimizer.compute_gradients(
          vars_[0] * vars_[0] + vars_[1] + vars_[2], vars_)

      variables.global_variables_initializer().run()
      expect_grads = ([4], [1], [1])
      index = 0
      for grad, _ in grads_and_vars:
        self.assertAllCloseAccordingToType(expect_grads[index],
                                           self.evaluate(grad))
        index += 1

  @test_util.deprecated_graph_mode_only
  def testMapGrandientDescentWithSquare2(self):
    #test with map_fn_quadratic(), x*y + x*z + y*z
    with self.cached_session():
      optimizer = gd.GradientDescentOptimizer(3.0)
      values = [1.0, 2.0, 3.0]
      vars_ = [variables.Variable([v], dtype=dtypes.float32) for v in values]
      map_optimizer = map_gradient_optimizer.MapGradientOptimizer(
          optimizer, map_fn_quadratic)
      grads_and_vars = map_optimizer.compute_gradients(
          vars_[0] * vars_[1] + vars_[0] * vars_[2] + vars_[1] * vars_[2],
          vars_)

      variables.global_variables_initializer().run()
      expect_grads = ([25], [16], [9])
      index = 0
      for grad, _ in grads_and_vars:
        self.assertAllCloseAccordingToType(expect_grads[index],
                                           self.evaluate(grad))
        index += 1

  @test_util.deprecated_graph_mode_only
  def testLambda(self):
    #test with lambda, x*y + x*z + y*z
    with self.cached_session():
      optimizer = gd.GradientDescentOptimizer(3.0)
      values = [1.0, 2.0, 3.0]
      vars_ = [variables.Variable([v], dtype=dtypes.float32) for v in values]
      map_optimizer = map_gradient_optimizer.MapGradientOptimizer(
          optimizer, lambda grad_lamb, var_lamb: math_ops.square(grad_lamb))
      grads_and_vars = map_optimizer.compute_gradients(
          vars_[0] * vars_[1] + vars_[0] * vars_[2] + vars_[1] * vars_[2],
          vars_)

      variables.global_variables_initializer().run()
      expect_grads = ([25], [16], [9])
      index = 0
      for grad, _ in grads_and_vars:
        self.assertAllCloseAccordingToType(expect_grads[index],
                                           self.evaluate(grad))
        index += 1

  @test_util.deprecated_graph_mode_only
  def testClipGradientOptimizer(self):
    with self.cached_session():
      optimizer = gd.GradientDescentOptimizer(3.0)
      values = [1, 5, 10]
      vars_ = [variables.Variable([v], dtype=dtypes.float32) for v in values]
      map_optimizer = map_gradient_optimizer.MapGradientOptimizer(
          optimizer, map_fn_clipping_7_and_14)
      grads_and_vars = map_optimizer.compute_gradients(
          vars_[0] * vars_[1] + vars_[0] * vars_[2] + vars_[1] * vars_[2],
          vars_)
      variables.global_variables_initializer().run()
      expect_grads = ([14], [11], [7])
      index = 0
      for grad, _ in grads_and_vars:
        self.assertAllCloseAccordingToType(expect_grads[index],
                                           self.evaluate(grad))
        index += 1

  @test_util.deprecated_graph_mode_only
  def testWeightDecay(self):
    with self.cached_session():
      optimizer = gd.GradientDescentOptimizer(3.0)
      values = [1, 5, 10]
      vars_ = [variables.Variable([v], dtype=dtypes.float32) for v in values]
      map_optimizer = map_gradient_optimizer.MapGradientOptimizer(
          optimizer, map_fn_decay)
      grads_and_vars = map_optimizer.compute_gradients(
          vars_[0] * vars_[1] + vars_[0] * vars_[2] + vars_[1] * vars_[2],
          vars_)
      variables.global_variables_initializer().run()
      expect_grads = ([15.01], [11.05], [6.1])
      index = 0
      for grad, _ in grads_and_vars:
        self.assertAllCloseAccordingToType(expect_grads[index],
                                           self.evaluate(grad))
        index += 1

  @test_util.deprecated_graph_mode_only
  def testMinimize(self):
    with self.cached_session():
      values = [1, 2, 3]
      vars_ = [variables.Variable([v], dtype=dtypes.float32) for v in values]

      optimizer = gd.GradientDescentOptimizer(1.0)
      map_optimizer = map_gradient_optimizer.MapGradientOptimizer(
          optimizer, map_fn_quadratic)
      loss = math_ops.reduce_prod(vars_)
      train_op = map_optimizer.minimize(loss, var_list=vars_)
      variables.global_variables_initializer().run()
      train_op.run()

      # Loss is a*b*c
      # so dL/dV = [v1*v2,v0*v2,v0*v1] = 6, 3, 2
      # Which is then squared with the MapOptimizer = 36, 9, 4
      # Grads then applied to the weights via GD w/ learning rate of 1.
      # = -35, -7, -1
      expect_weights = ([-35.0], [-7.0], [-1.0])
      for expected_weight, actual_weight_tensor in zip(expect_weights, vars_):
        self.assertAllCloseAccordingToType(expected_weight,
                                           self.evaluate(actual_weight_tensor))


if __name__ == "__main__":
  googletest.main()
