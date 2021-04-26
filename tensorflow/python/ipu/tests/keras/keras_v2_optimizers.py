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

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python import keras
from tensorflow.python import ipu
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ops import math_ops
from tensorflow.python.ipu.keras.optimizers import CrossReplicaOptimizer
from tensorflow.python.ipu.keras.optimizers import MapGradientOptimizer
from tensorflow.python.ipu.keras.optimizers import IpuOptimizer
from tensorflow.python.framework.constant_op import constant as tf_constant

v0 = 5
data_init = 2


def create_model(use_ipu, optimizer):
  Sequential = ipu.keras.Sequential if use_ipu else keras.Sequential
  one_init = keras.initializers.Constant(1)
  v0_init = keras.initializers.Constant(v0)
  m = Sequential([
      keras.layers.Dense(1,
                         use_bias=False,
                         kernel_initializer=v0_init,
                         bias_initializer=one_init)
  ])

  l = keras.losses.MeanSquaredError(reduction="sum_over_batch_size")
  m.compile(loss=l, optimizer=optimizer)
  return m


def map_fn_quadratic(grad, _):
  return math_ops.square(grad)


def map_fn_add(grad, _):
  h = tf_constant(([10.0]))
  return math_ops.add(grad, h)


nipus = 2
x = np.full((128, 1), data_init, np.single)
y = np.full((128, 1), data_init, np.single)
learning_rate = 0.1
original_optimizer = keras.optimizer_v2.gradient_descent.SGD(
    learning_rate=learning_rate)


class KerasV2OptimizersTest(test_util.TensorFlowTestCase):
  def get_model_weight(self, m):
    w = m.get_weights()
    self.assertEqual(len(w), 1)
    w = w[0]
    return w.item()

  # The cross replica optimizer is used specifically for IPU's to sum gradients
  # across the replicas. This should produce the exact same result as simply
  # summing across the batch with the unadjusted optimizer
  @tu.test_uses_ipus(num_ipus=nipus, allow_ipu_model=False)
  @test_util.run_v2_only
  def testCrossReplicaOptimizer(self):
    cross_replica_optimizer = CrossReplicaOptimizer(original_optimizer)

    cfg = ipu_utils.create_ipu_config()
    cfg = ipu_utils.auto_select_ipus(cfg, nipus)
    cfg = tu.add_hw_ci_connection_options(cfg)
    ipu_utils.configure_ipu_system(cfg)

    strategy = ipu_strategy.IPUStrategy()
    steps = 10
    batch_size = 2
    with strategy.scope():
      m = create_model(True, cross_replica_optimizer)
      m.fit(x=x, y=y, steps_per_epoch=steps, epochs=1, batch_size=batch_size)

    cpu_model = create_model(False, original_optimizer)
    cpu_model.fit(x=x,
                  y=y,
                  steps_per_epoch=steps / nipus,
                  epochs=1,
                  batch_size=nipus * batch_size)

    # re enable when T36442 is fixed
    #
    #self.assertEqual(m.get_weights(), cpu_model.get_weights())

  @test_util.run_v2_only
  def testMapGradientOptimizer(self):
    quad_optimizer = IpuOptimizer(
        MapGradientOptimizer(original_optimizer, map_fn_quadratic))
    cfg = ipu_utils.create_ipu_config()
    cfg = ipu_utils.auto_select_ipus(cfg, 1)
    cfg = tu.add_hw_ci_connection_options(cfg)
    ipu_utils.configure_ipu_system(cfg)
    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = create_model(True, quad_optimizer)
      m.fit(x=x, y=y, steps_per_epoch=1, epochs=1, batch_size=1)

    grad = (2 * data_init * ((v0 * data_init) - (data_init)))
    expected = v0 - (learning_rate * (grad**2))
    self.assertAllCloseAccordingToType(self.get_model_weight(m), expected)

  @test_util.run_v2_only
  def testMapGradientOptimizerNested(self):
    quad_optimizer = MapGradientOptimizer(
        MapGradientOptimizer(original_optimizer, map_fn_quadratic), map_fn_add)

    cfg = ipu_utils.create_ipu_config()
    cfg = ipu_utils.auto_select_ipus(cfg, 1)
    cfg = tu.add_hw_ci_connection_options(cfg)
    ipu_utils.configure_ipu_system(cfg)
    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = create_model(True, quad_optimizer)
      m.fit(x=x, y=y, steps_per_epoch=1, epochs=1, batch_size=1)

      grad = (2 * data_init * ((v0 * data_init) - (data_init)))
      expected = v0 - (learning_rate * ((grad**2) + 10))
      self.assertAllCloseAccordingToType(self.get_model_weight(m), expected)

  @tu.test_uses_ipus(num_ipus=nipus, allow_ipu_model=False)
  @test_util.run_v2_only
  def testMappedAndCross(self):
    # test that _keras optimizer wrapper still works with default optimizers
    add_optimizer = CrossReplicaOptimizer(
        MapGradientOptimizer(original_optimizer, map_fn_add))

    cfg = ipu_utils.create_ipu_config()
    cfg = ipu_utils.auto_select_ipus(cfg, nipus)
    cfg = tu.add_hw_ci_connection_options(cfg)
    ipu_utils.configure_ipu_system(cfg)
    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = create_model(True, add_optimizer)
      m.fit(x=x, y=y, steps_per_epoch=2, epochs=1, batch_size=1)

      #grad = (2 * data_init * ((v0 * data_init) - (data_init)))
      #expected = v0 - (learning_rate * (grad + 10))
      # re enable when T36442 is fixed
      #self.assertAllCloseAccordingToType(self.get_model_weight(m), expected)


if __name__ == "__main__":
  test.main()
