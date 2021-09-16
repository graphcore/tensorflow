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
# =============================================================================

import copy
import numpy as np
from absl.testing import parameterized

from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.eager import def_function
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.framework import test_util
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.keras.optimizers import AutomaticLossScalingOptimizer
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizer_v2 import adagrad as adagrad_v2
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import googletest

BATCH_SIZE = 8
INPUT_SHAPE = (BATCH_SIZE, 4)
OUTPUT_SHAPE = (BATCH_SIZE, 128)

DATA = np.ones(shape=INPUT_SHAPE, dtype=np.float32)
TARGETS = np.ones(shape=OUTPUT_SHAPE, dtype=np.float32)

OPTIMIZER_CASES = [{
    'testcase_name': 'Adagrad',
    'optimizer_type': adagrad_v2.Adagrad,
    'optimizer_args': [0.01],
    'optimizer_kwargs': {}
}, {
    'testcase_name': 'Adam',
    'optimizer_type': adam_v2.Adam,
    'optimizer_args': [0.01],
    'optimizer_kwargs': {}
}, {
    'testcase_name': 'GradientDescent',
    'optimizer_type': gradient_descent_v2.SGD,
    'optimizer_args': [0.01],
    'optimizer_kwargs': {},
}]

ALS_OPTIMIZER_KWARG_CASES = [{
    'initial_loss_scaling_factor': 10.0,
    'update_frequency': 2,
    'increase_factor': 2.0,
    'decrease_factor': 0.5
}, {
    'initial_loss_scaling_factor': 20.0,
    'update_frequency': 2,
    'increase_factor': 1.25,
    'decrease_factor': 0.75
}, {
    'initial_loss_scaling_factor': 30.0,
    'update_frequency': 2,
    'increase_factor': 1.33,
    'decrease_factor': 0.66
}, {
    'initial_loss_scaling_factor': 40.0,
    'update_frequency': 2,
    'increase_factor': 2.0,
    'decrease_factor': 0.5
}, {
    'initial_loss_scaling_factor': 50.0,
    'update_frequency': 2,
    'increase_factor': 1.1,
    'decrease_factor': 0.9
}, {
    'initial_loss_scaling_factor': 60.0,
    'update_frequency': 2,
    'increase_factor': 4.0,
    'decrease_factor': 0.25
}]


def generate_test_cases():
  cases = []
  for opt_case in OPTIMIZER_CASES:
    for n, als_case in enumerate(ALS_OPTIMIZER_KWARG_CASES):
      c = copy.deepcopy(opt_case)
      c['testcase_name'] += 'TestCase%d' % n
      c['als_kwargs'] = als_case
      cases.append(c)
  return cases


TEST_CASES = generate_test_cases()


class AutomaticLossScalingOptimizerTest(test_util.TensorFlowTestCase,
                                        parameterized.TestCase):
  def testInvalidInitialLSF(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(
        ValueError,
        "initial_loss_scaling_factor must be nonzero and positive"):
      _ = AutomaticLossScalingOptimizer(opt, initial_loss_scaling_factor=0.0)

  def testInvalidUpdateFrequency(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(
        ValueError, "update_frequency must be nonzero and positive"):
      _ = AutomaticLossScalingOptimizer(opt, update_frequency=0)

  def testInvalidIncreaseFactor(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(
        ValueError, "increase_factor must be nonzero and positive"):
      _ = AutomaticLossScalingOptimizer(opt, increase_factor=0)

  def testInvalidDecreaseFactor(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(
        ValueError, "decrease_factor must be nonzero and positive"):
      _ = AutomaticLossScalingOptimizer(opt, decrease_factor=0)

  def testInvalidIncreaseDecreaseFactors(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(
        ValueError, "decrease_factor must be less than increase_factor"):
      _ = AutomaticLossScalingOptimizer(opt,
                                        decrease_factor=2,
                                        increase_factor=1)

  def testSetTooHighDecreaseFactor(self):
    opt = gradient_descent_v2.SGD(0.1)
    opt_wrapper = AutomaticLossScalingOptimizer(opt,
                                                increase_factor=2.0,
                                                decrease_factor=0.5)
    with self.assertRaisesRegex(
        ValueError, "decrease_factor must be less than increase_factor"):
      opt_wrapper.decrease_factor = 3.0

  def testSetTooLowIncreaseFactor(self):
    opt = gradient_descent_v2.SGD(0.1)
    opt_wrapper = AutomaticLossScalingOptimizer(opt,
                                                increase_factor=2.0,
                                                decrease_factor=0.5)
    with self.assertRaisesRegex(
        ValueError, "increase_factor must be greater than decrease_factor"):
      opt_wrapper.increase_factor = 0.1

  def testCannotSetHistogram(self):
    opt = gradient_descent_v2.SGD(0.1)
    opt_wrapper = AutomaticLossScalingOptimizer(opt)
    with self.assertRaisesRegex(ValueError,
                                "histogram is a read only property."):
      opt_wrapper.histogram = None

  def testCannotSetNormalizedHistogram(self):
    opt = gradient_descent_v2.SGD(0.1)
    opt_wrapper = AutomaticLossScalingOptimizer(opt)
    with self.assertRaisesRegex(
        ValueError, "normalized_histogram is a read only property."):
      opt_wrapper.normalized_histogram = None

  def testCannotSetLSF(self):
    opt = gradient_descent_v2.SGD(0.1)
    opt_wrapper = AutomaticLossScalingOptimizer(opt)
    with self.assertRaisesRegex(
        ValueError, "loss_scaling_factor is a read only property."):
      opt_wrapper.loss_scaling_factor = 1

  def testCannotSetClipLevels(self):
    opt = gradient_descent_v2.SGD(0.1)
    opt_wrapper = AutomaticLossScalingOptimizer(opt)
    with self.assertRaisesRegex(ValueError,
                                "clip_levels is a read only property."):
      opt_wrapper.clip_levels = None

  def testInvalidMaxLSF(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(
        ValueError, "max_loss_scaling_factor must be greater than one"):
      _ = AutomaticLossScalingOptimizer(opt, max_loss_scaling_factor=1)

  def testRatioThresholdTooLow(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(
        ValueError,
        "ratio_threshold must be greater than zero and less than one"):
      _ = AutomaticLossScalingOptimizer(opt, ratio_threshold=-1)

  def testRatioThresholdTooHigh(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(
        ValueError,
        "ratio_threshold must be greater than zero and less than one"):
      _ = AutomaticLossScalingOptimizer(opt, ratio_threshold=2)

  def testInitialLSFTooHigh(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(
        ValueError, "initial_loss_scaling_factor must be less than"
        " max_loss_scaling_factor"):
      _ = AutomaticLossScalingOptimizer(opt,
                                        initial_loss_scaling_factor=4.0,
                                        max_loss_scaling_factor=3.0)

  def testInitialLSFAndIncreaseFactorTooHigh(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(
        ValueError,
        "initial_loss_scaling_factor x increase_factor must be less "
        "than max_loss_scaling_factor"):
      _ = AutomaticLossScalingOptimizer(opt,
                                        initial_loss_scaling_factor=1.0,
                                        increase_factor=4.0,
                                        max_loss_scaling_factor=4.0)

  @parameterized.named_parameters(*TEST_CASES)
  def testSimpleTraining(self, optimizer_type, optimizer_args,
                         optimizer_kwargs, als_kwargs):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      opt = optimizer_type(*optimizer_args, **optimizer_kwargs)
      opt_wrapper = AutomaticLossScalingOptimizer(opt, **als_kwargs)

      dense = layers.Dense(
          OUTPUT_SHAPE[1],
          activation='relu',
          dtype=np.float16,
          kernel_initializer=init_ops.constant_initializer(1.0))

      @def_function.function(experimental_compile=True)
      def f(x, t):
        with GradientTape() as tape:
          y = dense(x)
          l = losses.mean_squared_error(labels=t, predictions=y)

        opt_wrapper.minimize(l, dense.trainable_variables, tape=tape)
        return l

      model_losses = []
      for _ in range(3):
        res = strategy.run(f, args=[DATA, TARGETS])
        model_losses.append(res)

    last_loss = float('inf')
    for r in model_losses:
      self.assertTrue(np.isfinite(r))
      self.assertLess(r, last_loss)
      last_loss = r

  @parameterized.named_parameters(*TEST_CASES)
  def testDistributionWithSaturation(self, optimizer_type, optimizer_args,
                                     optimizer_kwargs, als_kwargs):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      opt = optimizer_type(*optimizer_args, **optimizer_kwargs)
      opt_wrapper = AutomaticLossScalingOptimizer(opt, **als_kwargs)

      dense0 = layers.Dense(
          OUTPUT_SHAPE[1],
          activation='relu',
          dtype=np.float16,
          kernel_initializer=init_ops.constant_initializer(1.0))

      dense1 = layers.Dense(
          OUTPUT_SHAPE[1],
          activation='relu',
          dtype=np.float32,
          kernel_initializer=init_ops.constant_initializer(1.0))

      @def_function.function(experimental_compile=True)
      def f(x, t):
        with GradientTape() as tape:
          y = dense1(dense0(x))
          l = losses.mean_squared_error(labels=t, predictions=y)

        v = dense0.trainable_variables + dense1.trainable_variables
        opt_wrapper.minimize(l, v, tape=tape)
        return l

      @def_function.function(experimental_compile=True)
      def g():
        return opt_wrapper.loss_scaling_factor

      @def_function.function(experimental_compile=True)
      def h():
        return opt_wrapper.histogram

      targets_huge = TARGETS * np.finfo(np.float16).max
      for _ in range(opt_wrapper.update_frequency - 1):
        l = strategy.run(f, args=[DATA, targets_huge])
        self.assertTrue(np.isfinite(l))

        # We expect gradients only for the float16 dense layer to be taken into
        # consideration in the histogram. In this case, most units gradients
        # in this layer should have overflowed.
        hist = strategy.run(h)
        self.assertGreater(hist[1], hist[0])

        # Check the LSF hasn't changed yet.
        lsf = strategy.run(g)
        self.assertAllEqual(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that the LSF decreases after the next epoch.
      _ = strategy.run(f, args=[DATA, targets_huge])
      lsf = strategy.run(g)
      self.assertLess(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that it's the expected value.
      expected_lsf = \
        opt_wrapper.initial_loss_scaling_factor * opt_wrapper.decrease_factor
      self.assertAllClose(lsf, expected_lsf)

      # Check that the histogram has been reset as there has
      # been an LSF update.
      hist = strategy.run(h)
      self.assertAllEqual(hist, np.zeros_like(hist))

  @parameterized.named_parameters(*TEST_CASES)
  def testDistributionWithoutSaturation(self, optimizer_type, optimizer_args,
                                        optimizer_kwargs, als_kwargs):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      opt = optimizer_type(*optimizer_args, **optimizer_kwargs)
      opt_wrapper = AutomaticLossScalingOptimizer(opt, **als_kwargs)

      dense0 = layers.Dense(
          OUTPUT_SHAPE[1],
          activation='relu',
          dtype=np.float16,
          kernel_initializer=init_ops.constant_initializer(1.0))

      dense1 = layers.Dense(
          OUTPUT_SHAPE[1],
          activation='relu',
          dtype=np.float32,
          kernel_initializer=init_ops.constant_initializer(1.0))

      @def_function.function(experimental_compile=True)
      def f(x, t):
        with GradientTape() as tape:
          y = dense1(dense0(x))
          l = losses.mean_squared_error(labels=t, predictions=y)

        v = dense0.trainable_variables + dense1.trainable_variables
        opt_wrapper.minimize(l, v, tape=tape)
        return l

      @def_function.function(experimental_compile=True)
      def g():
        return opt_wrapper.loss_scaling_factor

      @def_function.function(experimental_compile=True)
      def h():
        return opt_wrapper.histogram

      for _ in range(opt_wrapper.update_frequency - 1):
        l = strategy.run(f, args=[DATA, TARGETS])
        self.assertTrue(np.isfinite(l))

        # We expect gradients only for the float16 dense layer to be taken into
        # consideration in the histogram. In this case, most units gradients
        # in this layer should not have overflowed.
        hist = strategy.run(h)
        self.assertGreater(hist[0], hist[1])

        # Check the LSF hasn't changed yet.
        lsf = strategy.run(g)
        self.assertAllEqual(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that the LSF increases after the next epoch.
      _ = strategy.run(f, args=[DATA, TARGETS])
      lsf = strategy.run(g)
      self.assertGreater(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that it's the expected value.
      expected_lsf = \
        opt_wrapper.initial_loss_scaling_factor * opt_wrapper.increase_factor
      self.assertAllClose(lsf, expected_lsf)

      # Check that the histogram has been reset as there has
      # been an LSF update.
      hist = strategy.run(h)
      self.assertAllEqual(hist, np.zeros_like(hist))

  @parameterized.named_parameters(*TEST_CASES)
  def testDistributionWithSaturationNoStatAccum(self, optimizer_type,
                                                optimizer_args,
                                                optimizer_kwargs, als_kwargs):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      opt = optimizer_type(*optimizer_args, **optimizer_kwargs)
      opt_wrapper = AutomaticLossScalingOptimizer(
          opt, **als_kwargs, accumulate_statistics_over_update_period=False)

      dense0 = layers.Dense(
          OUTPUT_SHAPE[1],
          activation='relu',
          dtype=np.float16,
          kernel_initializer=init_ops.constant_initializer(1.0))

      dense1 = layers.Dense(
          OUTPUT_SHAPE[1],
          activation='relu',
          dtype=np.float32,
          kernel_initializer=init_ops.constant_initializer(1.0))

      @def_function.function(experimental_compile=True)
      def f(x, t):
        with GradientTape() as tape:
          y = dense1(dense0(x))
          l = losses.mean_squared_error(labels=t, predictions=y)

        v = dense0.trainable_variables + dense1.trainable_variables
        opt_wrapper.minimize(l, v, tape=tape)
        return l

      @def_function.function(experimental_compile=True)
      def g():
        return opt_wrapper.loss_scaling_factor

      @def_function.function(experimental_compile=True)
      def h():
        return opt_wrapper.histogram

      targets_huge = TARGETS * np.finfo(np.float16).max
      for _ in range(opt_wrapper.update_frequency - 1):
        l = strategy.run(f, args=[DATA, targets_huge])
        self.assertTrue(np.isfinite(l))

        # In this loop, we expect the histogram to always be zeros.
        hist = strategy.run(h)
        self.assertAllEqual(hist, np.zeros_like(hist))

        # Check the LSF hasn't changed yet.
        lsf = strategy.run(g)
        self.assertAllEqual(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that the LSF decreases after the next epoch.
      _ = strategy.run(f, args=[DATA, targets_huge])
      lsf = strategy.run(g)
      self.assertLess(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that it's the expected value.
      expected_lsf = \
        opt_wrapper.initial_loss_scaling_factor * opt_wrapper.decrease_factor
      self.assertAllClose(lsf, expected_lsf)

      # Check that the histogram has been reset as there has
      # been an LSF update.
      hist = strategy.run(h)
      self.assertAllEqual(hist, np.zeros_like(hist))

  @parameterized.named_parameters(*TEST_CASES)
  def testDistributionWithoutSaturationNoStatAccum(self, optimizer_type,
                                                   optimizer_args,
                                                   optimizer_kwargs,
                                                   als_kwargs):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      opt = optimizer_type(*optimizer_args, **optimizer_kwargs)
      opt_wrapper = AutomaticLossScalingOptimizer(
          opt, **als_kwargs, accumulate_statistics_over_update_period=False)

      dense0 = layers.Dense(
          OUTPUT_SHAPE[1],
          activation='relu',
          dtype=np.float16,
          kernel_initializer=init_ops.constant_initializer(1.0))

      dense1 = layers.Dense(
          OUTPUT_SHAPE[1],
          activation='relu',
          dtype=np.float32,
          kernel_initializer=init_ops.constant_initializer(1.0))

      @def_function.function(experimental_compile=True)
      def f(x, t):
        with GradientTape() as tape:
          y = dense1(dense0(x))
          l = losses.mean_squared_error(labels=t, predictions=y)

        v = dense0.trainable_variables + dense1.trainable_variables
        opt_wrapper.minimize(l, v, tape=tape)
        return l

      @def_function.function(experimental_compile=True)
      def g():
        return opt_wrapper.loss_scaling_factor

      @def_function.function(experimental_compile=True)
      def h():
        return opt_wrapper.histogram

      for _ in range(opt_wrapper.update_frequency - 1):
        l = strategy.run(f, args=[DATA, TARGETS])
        self.assertTrue(np.isfinite(l))

        # In this loop, we expect the histogram to always be zeros.
        hist = strategy.run(h)
        self.assertAllEqual(hist, np.zeros_like(hist))

        # Check the LSF hasn't changed yet.
        lsf = strategy.run(g)
        self.assertAllEqual(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that the LSF increases after the next epoch.
      _ = strategy.run(f, args=[DATA, TARGETS])
      lsf = strategy.run(g)
      self.assertGreater(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that it's the expected value.
      expected_lsf = \
        opt_wrapper.initial_loss_scaling_factor * opt_wrapper.increase_factor
      self.assertAllClose(lsf, expected_lsf)

      # Check that the histogram has been reset as there has
      # been an LSF update.
      hist = strategy.run(h)
      self.assertAllEqual(hist, np.zeros_like(hist))

  @parameterized.named_parameters(*TEST_CASES)
  def testSimpleTrainingKeras(self, optimizer_type, optimizer_args,
                              optimizer_kwargs, als_kwargs):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      opt = optimizer_type(*optimizer_args, **optimizer_kwargs)
      opt_wrapper = AutomaticLossScalingOptimizer(opt, **als_kwargs)

      input_layer = keras.layers.Input(shape=DATA.shape[1])

      dense = layers.Dense(
          OUTPUT_SHAPE[1],
          activation='relu',
          dtype=np.float16,
          kernel_initializer=init_ops.constant_initializer(1.0))(input_layer)

      m = keras.Model(inputs=input_layer, outputs=dense)
      m.compile(optimizer=opt_wrapper, loss='mse')

      history = m.fit(DATA, TARGETS, epochs=3, batch_size=1, verbose=False)

    last_loss = float('inf')
    for l in history.history['loss']:
      self.assertTrue(np.isfinite(l))
      self.assertLess(l, last_loss)
      last_loss = l

  @parameterized.named_parameters(*TEST_CASES)
  def testDistributionWithSaturationKeras(self, optimizer_type, optimizer_args,
                                          optimizer_kwargs, als_kwargs):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      opt = optimizer_type(*optimizer_args, **optimizer_kwargs)
      opt_wrapper = AutomaticLossScalingOptimizer(opt, **als_kwargs)

      input_layer = keras.layers.Input(shape=DATA.shape[1])

      dense0 = layers.Dense(
          OUTPUT_SHAPE[1],
          activation='relu',
          dtype=np.float16,
          kernel_initializer=init_ops.constant_initializer(1.0))(input_layer)

      dense1 = layers.Dense(
          OUTPUT_SHAPE[1],
          activation='relu',
          dtype=np.float32,
          kernel_initializer=init_ops.constant_initializer(1.0))(dense0)

      m = keras.Model(inputs=input_layer, outputs=dense1)
      m.compile(optimizer=opt_wrapper, loss='mse')

      @def_function.function(experimental_compile=True)
      def g():
        return opt_wrapper.loss_scaling_factor

      @def_function.function(experimental_compile=True)
      def h():
        return opt_wrapper.histogram

      targets_huge = TARGETS * np.finfo(np.float16).max
      for _ in range(opt_wrapper.update_frequency - 1):
        history = m.fit(DATA,
                        targets_huge,
                        epochs=1,
                        batch_size=BATCH_SIZE,
                        verbose=False)

        self.assertTrue(np.isfinite(history.history['loss'][0]))

        # We expect gradients only for the float16 dense layer to be taken into
        # consideration in the histogram. In this case, most units gradients
        # in this layer should have overflowed.
        hist = strategy.run(h)
        self.assertGreater(hist[1], hist[0])

        # Check the LSF hasn't changed yet.
        lsf = strategy.run(g)
        self.assertAllEqual(lsf, opt_wrapper.initial_loss_scaling_factor)

      # The LSF should have decreased after this next epoch.
      history = m.fit(DATA,
                      targets_huge,
                      epochs=1,
                      batch_size=BATCH_SIZE,
                      verbose=False)

      lsf = strategy.run(g)
      self.assertLess(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that it's the expected value.
      expected_lsf = \
        opt_wrapper.initial_loss_scaling_factor * opt_wrapper.decrease_factor
      self.assertAllClose(lsf, expected_lsf)

      # Check that the histogram has been reset as there has
      # been an LSF update.
      hist = strategy.run(h)
      self.assertAllEqual(hist, np.zeros_like(hist))

  @parameterized.named_parameters(*TEST_CASES)
  def testDistributionWithoutSaturationKeras(self, optimizer_type,
                                             optimizer_args, optimizer_kwargs,
                                             als_kwargs):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      opt = optimizer_type(*optimizer_args, **optimizer_kwargs)
      opt_wrapper = AutomaticLossScalingOptimizer(opt, **als_kwargs)

      input_layer = keras.layers.Input(shape=DATA.shape[1])

      dense0 = layers.Dense(
          OUTPUT_SHAPE[1],
          activation='relu',
          dtype=np.float16,
          kernel_initializer=init_ops.constant_initializer(1.0))(input_layer)

      dense1 = layers.Dense(
          OUTPUT_SHAPE[1],
          activation='relu',
          dtype=np.float32,
          kernel_initializer=init_ops.constant_initializer(1.0))(dense0)

      m = keras.Model(inputs=input_layer, outputs=dense1)
      m.compile(optimizer=opt_wrapper, loss='mse')

      @def_function.function(experimental_compile=True)
      def g():
        return opt_wrapper.loss_scaling_factor

      @def_function.function(experimental_compile=True)
      def h():
        return opt_wrapper.histogram

      for _ in range(opt_wrapper.update_frequency - 1):
        history = m.fit(DATA,
                        TARGETS,
                        epochs=1,
                        batch_size=BATCH_SIZE,
                        verbose=False)

        self.assertTrue(np.isfinite(history.history['loss'][0]))

        # We expect gradients only for the float16 dense layer to be taken into
        # consideration in the histogram. In this case, most units gradients
        # in this layer should not have overflowed.
        hist = strategy.run(h)
        self.assertGreater(hist[0], hist[1])

        # Check the LSF hasn't changed yet.
        lsf = strategy.run(g)
        self.assertAllEqual(lsf, opt_wrapper.initial_loss_scaling_factor)

      # The LSF should have increased after this next epoch.
      history = m.fit(DATA,
                      TARGETS,
                      epochs=1,
                      batch_size=BATCH_SIZE,
                      verbose=False)

      lsf = strategy.run(g)
      self.assertGreater(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that it's the expected value.
      expected_lsf = \
        opt_wrapper.initial_loss_scaling_factor * opt_wrapper.increase_factor
      self.assertAllClose(lsf, expected_lsf)

      # Check that the histogram has been reset as there has
      # been an LSF update.
      hist = strategy.run(h)
      self.assertAllEqual(hist, np.zeros_like(hist))

  def testUpperLSFCap(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      opt = gradient_descent_v2.SGD(0.01)
      opt_wrapper = AutomaticLossScalingOptimizer(
          opt,
          increase_factor=2.0,
          decrease_factor=0.5,
          initial_loss_scaling_factor=1.0,
          max_loss_scaling_factor=16.0,
          update_frequency=1)

      v = variables.Variable(1.0, dtype=np.float16)

      @def_function.function(experimental_compile=True)
      def f():
        with GradientTape() as tape:
          y = 3 * v
          l = losses.mean_squared_error(labels=array_ops.ones_like(v),
                                        predictions=y)
        opt_wrapper.minimize(l, [v], tape=tape)
        return l

      @def_function.function(experimental_compile=True)
      def g():
        return opt_wrapper.loss_scaling_factor

      @def_function.function(experimental_compile=True)
      def h():
        return v.assign(1.0)

      # We expect the LSF to increase and cap at 8.
      for expected_lsf in [2.0, 4.0, 8.0, 8.0]:
        # "Train"
        _ = strategy.run(f)

        # Check the LSF.
        lsf = strategy.run(g)
        self.assertAllEqual(lsf, expected_lsf)

        # Reset var.
        _ = strategy.run(h)

  def testLowerLSFCap(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      opt = gradient_descent_v2.SGD(0.01)
      opt_wrapper = AutomaticLossScalingOptimizer(
          opt,
          increase_factor=2.0,
          decrease_factor=0.5,
          initial_loss_scaling_factor=16.0,
          update_frequency=1)

      v_init = np.finfo(np.float16).max
      v = variables.Variable(v_init, dtype=np.float16, shape=())

      @def_function.function(experimental_compile=True)
      def f():
        with GradientTape() as tape:
          y = 3 * v
          l = losses.mean_squared_error(labels=array_ops.ones_like(v),
                                        predictions=y)
        opt_wrapper.minimize(l, [v], tape=tape)
        return l

      @def_function.function(experimental_compile=True)
      def g():
        return opt_wrapper.loss_scaling_factor

      @def_function.function(experimental_compile=True)
      def h():
        return v.assign(v_init)

      # We expect the LSF to decrease and cap at 1.
      for expected_lsf in [8.0, 4.0, 2.0, 1.0, 1.0]:
        # "Train"
        _ = strategy.run(f)

        # Check the LSF.
        lsf = strategy.run(g)
        self.assertAllEqual(lsf, expected_lsf)

        # Reset var.
        _ = strategy.run(h)


if __name__ == "__main__":
  googletest.main()
