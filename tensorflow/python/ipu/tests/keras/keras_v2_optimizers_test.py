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

import unittest

from functools import partial

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python import keras
from tensorflow.python import ipu
from tensorflow.python.ipu import loops
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ops import math_ops
from tensorflow.python.ipu.keras.optimizers import CrossReplicaOptimizer
from tensorflow.python.ipu.keras.optimizers import MapGradientOptimizerInvertedChaining as MGOIC
from tensorflow.python.ipu.keras.optimizers import IpuOptimizer
from tensorflow.python.framework.constant_op import constant as tf_constant
from tensorflow.python.ipu.keras.optimizers import GradientAccumulationOptimizer
from tensorflow.python.keras.optimizer_v2 import gradient_descent

NUM_IPUS = 2
SGD_LEARNING_RATE = 0.01
NUM_SAMPLES = 128
INITIAL_WEIGHT_VALUE = 5.0
NUM_DENSE_UNITS = 1
DATA_VALUE = 2.0


def data_fn():
  return [np.full((NUM_SAMPLES, 1), DATA_VALUE, np.single)] * 2


def map_fn_quadratic(grad, _):
  return math_ops.square(grad)


def map_fn_add(grad, _):
  h = tf_constant(([10.0]))
  return math_ops.add(grad, h)


def map_fn_divide(grad, _):
  h = tf_constant([2.0])
  return math_ops.divide(grad, h)


def sgd():
  return gradient_descent.SGD(SGD_LEARNING_RATE)


def cross_replica_opt_fn():
  return CrossReplicaOptimizer(sgd())


def mgoic_opt_fn(f):
  return MGOIC(sgd(), f)


def dense_layer_fn():
  return keras.layers.Dense(
      NUM_DENSE_UNITS,
      use_bias=False,
      kernel_initializer=keras.initializers.Constant(INITIAL_WEIGHT_VALUE))


def sequential_model_fn(optimizer_fn, num_update_steps=1):
  m = keras.Sequential([dense_layer_fn()])
  l = keras.losses.MeanSquaredError(reduction="sum")
  m.compile(loss=l,
            optimizer=optimizer_fn(),
            steps_per_execution=num_update_steps)

  return m


TEST_CASES = [{
    'testcase_name': 'CrossReplicaOptimizer',
    'optimizer_fn': cross_replica_opt_fn,
}, {
    'testcase_name': 'MGOICAdd',
    'optimizer_fn': partial(mgoic_opt_fn, map_fn_add),
}]


class KerasV2OptimizersTest(test_util.TensorFlowTestCase,
                            parameterized.TestCase):
  def get_model_weight(self, m):
    w = m.get_weights()
    self.assertEqual(len(w), 1)
    w = w[0]
    return w.item()

  def verify_loss_decreases(self, losses):
    self.assertGreater(len(losses), 1)

    losses.reverse()
    last_loss = losses[0]
    for l in losses[1:]:
      self.assertLess(l, last_loss)

  # The cross replica optimizer is used specifically for IPU's to sum gradients
  # across the replicas. This should produce the exact same result as simply
  # summing across the batch with the unadjusted optimizer
  @unittest.skip("Test does not pass internally.")
  @tu.test_uses_ipus(num_ipus=NUM_IPUS, allow_ipu_model=False)
  @test_util.run_v2_only
  def testCrossReplicaOptimizer(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = NUM_IPUS
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    steps = 10
    batch_size = 2
    with strategy.scope():
      m = sequential_model_fn(cross_replica_opt_fn, 2)
      m.fit(*data_fn(), steps_per_epoch=steps, epochs=1, batch_size=batch_size)

    cpu_model = sequential_model_fn(cross_replica_opt_fn)
    cpu_model.fit(*data_fn(),
                  steps_per_epoch=steps / NUM_IPUS,
                  epochs=1,
                  batch_size=NUM_IPUS * batch_size)
    self.assertEqual(m.get_weights(), cpu_model.get_weights())

  @test_util.run_v2_only
  def testMapGradientOptimizer(self):
    for quad_optimizer in [
        lambda: IpuOptimizer(MGOIC(sgd(), map_fn_quadratic)),
        lambda: MGOIC(sgd(), map_fn_quadratic)
    ]:
      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()
      strategy = ipu_strategy.IPUStrategyV1()
      with strategy.scope():
        m = sequential_model_fn(quad_optimizer)
        m.fit(*data_fn(),
              steps_per_epoch=1,
              epochs=1,
              batch_size=1,
              verbose=False)

        grad = (2 * DATA_VALUE * ((INITIAL_WEIGHT_VALUE * DATA_VALUE) -
                                  (DATA_VALUE)))
        expected = INITIAL_WEIGHT_VALUE - (SGD_LEARNING_RATE * (grad**2))
        self.assertAllCloseAccordingToType(self.get_model_weight(m), expected)

  @test_util.run_v2_only
  def testMapGradientOptimizerNested(self):
    for quad_optimizer in [
        lambda: MGOIC(MGOIC(sgd(), map_fn_add), map_fn_quadratic)
    ]:
      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()
      strategy = ipu_strategy.IPUStrategyV1()
      with strategy.scope():
        m = sequential_model_fn(quad_optimizer)
        m.fit(*data_fn(),
              steps_per_epoch=1,
              epochs=1,
              batch_size=1,
              verbose=False)

        grad = (2 * DATA_VALUE * ((INITIAL_WEIGHT_VALUE * DATA_VALUE) -
                                  (DATA_VALUE)))
        expected = INITIAL_WEIGHT_VALUE - (SGD_LEARNING_RATE *
                                           ((grad**2) + 10))
        self.assertAllCloseAccordingToType(self.get_model_weight(m), expected)

  @unittest.skip("T42094 - MapGradientOptimizer needs fixing.")
  @tu.test_uses_ipus(num_ipus=NUM_IPUS, allow_ipu_model=False)
  @test_util.run_v2_only
  def testMappedAndCross(self):
    # test that _keras optimizer wrapper still works with default optimizers
    add_optimizer = CrossReplicaOptimizer(MGOIC(sgd(), map_fn_add))

    cfg = IPUConfig()
    cfg.auto_select_ipus = NUM_IPUS
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()
    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = sequential_model_fn(lambda: add_optimizer, 2)
      m.fit(*data_fn(), steps_per_epoch=2, epochs=1, batch_size=1)

      #grad = (2 * DATA_VALUE * ((INITIAL_WEIGHT_VALUE * DATA_VALUE) - (DATA_VALUE)))
      #expected = INITIAL_WEIGHT_VALUE - (SGD_LEARNING_RATE * (grad + 10))
      # re enable when T36442 is fixed
      #self.assertAllCloseAccordingToType(self.get_model_weight(m), expected)

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.run_v2_only
  def testCreateFromConfig(self, optimizer_fn, **kwargs):  # pylint: disable=unused-argument
    opt_1 = optimizer_fn()
    opt_1_config = opt_1.get_config()

    opt_2_config = opt_1_config.copy()
    opt_2_config['name'] += "_copy"

    opt_2 = opt_1.__class__.from_config(opt_2_config)
    self.assertEqual(opt_2.get_config(), opt_2_config)

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.run_v2_only
  def testWeightsPropertyRead(self, optimizer_fn, **kwargs):  # pylint: disable=unused-argument
    opt = optimizer_fn()
    w = opt.weights
    opt.set_weights(2 * w)
    self.assertEqual(opt.weights, 2 * w)

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.run_v2_only
  def testWeightsPropertyWrite(self, optimizer_fn, **kwargs):  # pylint: disable=unused-argument
    opt = optimizer_fn()
    with self.assertRaisesRegex(AttributeError, "can't set attribute"):
      opt.weights = 1

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.run_v2_only
  def testClipnormProperty(self, optimizer_fn, **kwargs):  # pylint: disable=unused-argument
    opt = optimizer_fn()
    if not opt.clipnorm:
      opt.clipnorm = 1

    clip_norm_val = opt.clipnorm
    opt.clipnorm = 2 * clip_norm_val
    self.assertEqual(opt.clipnorm, 2 * clip_norm_val)

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.run_v2_only
  def testGlobalClipnormProperty(self, optimizer_fn, **kwargs):  # pylint: disable=unused-argument
    opt = optimizer_fn()
    if not opt.global_clipnorm:
      opt.global_clipnorm = 1

    clip_norm_val = opt.global_clipnorm
    opt.global_clipnorm = 2 * clip_norm_val
    self.assertEqual(opt.global_clipnorm, 2 * clip_norm_val)

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.run_v2_only
  def testClipvalueProperty(self, optimizer_fn, **kwargs):  # pylint: disable=unused-argument
    opt = optimizer_fn()
    if not opt.clipvalue:
      opt.clipvalue = 1

    clip_val = opt.clipvalue
    opt.clipvalue = 2 * clip_val
    self.assertEqual(opt.clipvalue, 2 * clip_val)

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.run_v2_only
  def testVariablesMethod(self, optimizer_fn, **kwargs):  # pylint: disable=unused-argument
    opt = optimizer_fn()
    self.assertEqual(opt.get_weights(), opt.variables())

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.run_v2_only
  def testGetSetWeights(self, optimizer_fn, **kwargs):  # pylint: disable=unused-argument
    opt_1 = optimizer_fn()
    opt_2 = optimizer_fn()

    opt_2.set_weights([w * 2 for w in opt_1.get_weights()])

    for a, b in zip(opt_1.get_weights(), opt_2.get_weights()):
      self.assertEqual(b, 2 * a)

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.run_v2_only
  def testMinimizeWithGradientTape(self, optimizer_fn, num_update_steps=1):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      layer = dense_layer_fn()
      optimizer = optimizer_fn()
      loss = keras.losses.MeanSquaredError(reduction="sum")

      @def_function.function(experimental_compile=True)
      def f(a, t, _):
        with GradientTape() as tape:
          z = layer(a)
          l = loss(z, t)

          def ll():
            return l

        optimizer.minimize(ll, layer.trainable_variables, tape=tape)
        return a, t, l

      @def_function.function(experimental_compile=True)
      def g(a, t):
        _, _, l = loops.repeat(num_update_steps, f, inputs=[a, t, 0.0])
        return l

      losses = [strategy.run(g, args=data_fn()) for _ in range(3)]
      self.verify_loss_decreases(losses)

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.run_v2_only
  def testMinimizeWithoutGradientTape(self, optimizer_fn, num_update_steps=1):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      layer = dense_layer_fn()
      optimizer = optimizer_fn()
      loss = keras.losses.MeanSquaredError(reduction="sum")

      @def_function.function(experimental_compile=True)
      def f(a, t, _):
        def l():
          z = layer(a)
          return loss(z, t)

        ll = l()
        optimizer.minimize(l, layer.trainable_variables)
        return a, t, ll

      @def_function.function(experimental_compile=True)
      def g(a, t):
        _, _, l = loops.repeat(num_update_steps, f, inputs=[a, t, 0.0])
        return l

      losses = [strategy.run(g, args=data_fn()) for _ in range(3)]
      self.verify_loss_decreases(losses)

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.run_v2_only
  def testKerasSequentialModelTrain(self, optimizer_fn, num_update_steps=1):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = sequential_model_fn(optimizer_fn, num_update_steps)

      history = m.fit(*data_fn(), epochs=3, verbose=False)
      losses = [l for l in history.history['loss']]
      self.verify_loss_decreases(losses)

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.run_v2_only
  def testKerasFunctionalModelTrain(self, optimizer_fn, num_update_steps=1):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(1)
      x = dense_layer_fn()(input_layer)
      l = keras.losses.MeanSquaredError(reduction="sum")

      m = keras.Model(inputs=input_layer, outputs=x)
      m.compile(loss=l,
                optimizer=optimizer_fn(),
                steps_per_execution=num_update_steps)

      history = m.fit(*data_fn(), epochs=3, verbose=False)
      losses = [l for l in history.history['loss']]
      self.verify_loss_decreases(losses)

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.run_v2_only
  def testKerasSequentialPipelineTrain(self, optimizer_fn, **kwargs):  # pylint: disable=unused-argument
    if isinstance(optimizer_fn(), GradientAccumulationOptimizer):
      return

    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = keras.Sequential([
          dense_layer_fn(),  # Stage 0
          dense_layer_fn(),  # Stage 0.
          dense_layer_fn(),  # Stage 1.
          dense_layer_fn(),  # Stage 1.
      ])

      m.set_pipelining_options(gradient_accumulation_steps_per_replica=4,
                               experimental_normalize_gradients=True)
      m.set_pipeline_stage_assignment([0, 0, 1, 1])
      m.compile(optimizer_fn(), loss='mse', steps_per_execution=8)

      history = m.fit(*data_fn(), epochs=3, verbose=False)
      losses = [l for l in history.history['loss']]
      self.verify_loss_decreases(losses)

  @parameterized.named_parameters(*TEST_CASES)
  @test_util.run_v2_only
  def testKerasFunctionalPipelineTrain(self, optimizer_fn, **kwargs):  # pylint: disable=unused-argument
    if isinstance(optimizer_fn(), GradientAccumulationOptimizer):
      return

    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(1)

      with ipu.keras.PipelineStage(0):
        x = dense_layer_fn()(input_layer)
        x = dense_layer_fn()(x)

      with ipu.keras.PipelineStage(1):
        x = dense_layer_fn()(x)
        x = dense_layer_fn()(x)

      m = keras.Model(inputs=input_layer, outputs=x)
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=4,
                               experimental_normalize_gradients=True)
      m.compile(optimizer_fn(), loss='mse', steps_per_execution=8)

      history = m.fit(*data_fn(), epochs=3, verbose=False)
      losses = [l for l in history.history['loss']]
      self.verify_loss_decreases(losses)


if __name__ == "__main__":
  test.main()
