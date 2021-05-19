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
import numpy as np
from tensorflow.python.ipu.config import IPUConfig

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import googletest
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import keras as ipu_keras


def test_dataset(length=None, batch_size=1, x_val=1.0, y_val=0.2):
  constant_d = constant_op.constant(x_val, shape=[32])
  constant_l = constant_op.constant(y_val, shape=[2])

  ds = dataset_ops.Dataset.from_tensors((constant_d, constant_l))
  ds = ds.repeat(length)
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


def simple_model():
  return [
      keras.layers.Dense(4),
      keras.layers.Dense(8),
  ]


class KerasModelExecutionParametersTest(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=1)
  @test_util.run_v2_only
  def testMismatchDatasetLengthToAccumulationCount(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu_keras.Sequential(simple_model(), gradient_accumulation_count=64)
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError, "Sequential requires the number of mini-batches"
          " in the dataset .* accumulation count "):
        m.fit(test_dataset(length=32))

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.run_v2_only
  def testMismatchDatasetLengthToStepsPerRun(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu_keras.Sequential(simple_model(), gradient_accumulation_count=1)
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError, "Sequential requires the number of mini-batches in the"
          " dataset .* 'steps_per_run' "):
        m.fit(test_dataset(length=32), steps_per_run=64)

  @tu.test_uses_ipus(num_ipus=8)
  @test_util.run_v2_only
  def testMismatchDatasetLengthToReplicationFactor(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 8
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu_keras.Sequential(simple_model(), gradient_accumulation_count=1)
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError, "Sequential requires the number of mini-batches in the"
          " dataset .* replication factor "):
        m.fit(test_dataset(length=4), steps_per_run=1)

  @tu.test_uses_ipus(num_ipus=8)
  @test_util.run_v2_only
  def testMismatchDatasetLengthToValueCombination(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 8
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    # Only missmatched when considering all three of:
    # gradient_accumulation_count,
    # replication_factor,
    # and steps_per_run: (8 x 8 x 8) > 64
    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu_keras.Sequential(simple_model(), gradient_accumulation_count=8)
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError, "Sequential requires the number of mini-batches"
          " in the dataset "):
        m.fit(test_dataset(length=8), steps_per_run=8)

  @tu.test_uses_ipus(num_ipus=1)
  @test_util.run_v2_only
  def testMismatchStepsPerEpochToStepsPerRun(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu_keras.Sequential(simple_model())
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError, "Sequential requires the number of steps in " +
          "an epoch 'steps_per_epoch' "):
        m.fit(test_dataset(length=24), steps_per_epoch=24, steps_per_run=5)

  @tu.test_uses_ipus(num_ipus=8)
  @test_util.run_v2_only
  def testMismatchStepsPerEpochToReplicationFactor(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 8
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu_keras.Sequential(simple_model())
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError, "Sequential requires the number of steps in " +
          "an epoch 'steps_per_epoch' "):
        m.fit(test_dataset(length=20), steps_per_epoch=20, steps_per_run=1)

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.run_v2_only
  def testMismatchStepsPerEpochToValueCombination(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    # Only missmatched when considering both steps_per_run, and
    # replication_factor: 24 % (4 * 4) != 0
    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu_keras.Sequential(simple_model())
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError, "Sequential requires the number of steps in " +
          "an epoch 'steps_per_epoch' "):
        m.fit(test_dataset(length=24), steps_per_epoch=24, steps_per_run=4)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.run_v2_only
  def testPredictWithNumpyDataMismatchedStepsandReplicas(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()

    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32),
                                       dtype=np.single,
                                       batch_size=2)
      init = keras.initializers.Constant(0.1)
      x = keras.layers.Dense(4, name="layer0",
                             kernel_initializer=init)(input_layer)
      x = keras.layers.Dense(2, name="layer1", kernel_initializer=init)(x)
      m = ipu_keras.Model(input_layer, x, gradient_accumulation_count=3)

      m.compile('sgd', loss='mse')

      # Input data
      input_x = np.full([33, 32], 1.0, dtype=np.single)

      # Generate predictions.
      with self.assertRaisesRegex(
          ValueError,
          r"requires the number of steps 'steps' \(3\) to be evenly"
          r" divisible by the replication factor \(2\)"):
        m.predict(input_x, batch_size=2, steps=3)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.run_v2_only
  def testPredictWithNumpyDataMismatchStepsAndStepsPerRunAndReplicas(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()

    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32),
                                       dtype=np.single,
                                       batch_size=2)
      init = keras.initializers.Constant(0.1)
      x = keras.layers.Dense(4, name="layer0",
                             kernel_initializer=init)(input_layer)
      x = keras.layers.Dense(2, name="layer1", kernel_initializer=init)(x)
      m = ipu_keras.Model(input_layer, x, gradient_accumulation_count=3)

      m.compile('sgd', loss='mse')

      # Input data
      input_x = np.full([33, 32], 1.0, dtype=np.single)

      # Generate predictions.
      with self.assertRaisesRegex(
          ValueError,
          r"requires the number of steps 'steps' \(4\) to be evenly"
          r" divisible by the number of steps per execution of the"
          r" on-device inference loop 'steps_per_run' \(3\)"
          r" multiplied by the replication factor \(2\)."):
        m.predict(input_x, batch_size=2, steps=4, steps_per_run=3)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.run_v2_only
  def testEvaluateWithNumpyDataMismatchStepsAndStepsPerRunAndReplicas(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()

    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32),
                                       dtype=np.single,
                                       batch_size=2)
      init = keras.initializers.Constant(0.1)
      x = keras.layers.Dense(4, name="layer0",
                             kernel_initializer=init)(input_layer)
      x = keras.layers.Dense(2, name="layer1", kernel_initializer=init)(x)
      m = ipu_keras.Model(input_layer, x, gradient_accumulation_count=3)

      m.compile('sgd', loss='mse')

      # Input data
      input_x = np.full([33, 32], 1.0, dtype=np.single)
      input_y = np.full([33, 1], 1.0, dtype=np.single)

      # Run evaluation
      with self.assertRaisesRegex(
          ValueError,
          r"requires the number of steps 'steps' \(4\) to be evenly"
          r" divisible by the number of steps per execution of the"
          r" on-device evaluation loop 'steps_per_run' \(3\)"
          r" multiplied by the replication factor \(2\)."):
        m.evaluate(input_x, input_y, batch_size=2, steps=4, steps_per_run=3)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.run_v2_only
  def testFitWithNumpyDataMismatchStepsAndStepsPerRunAndReplicas(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()

    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32),
                                       dtype=np.single,
                                       batch_size=2)
      init = keras.initializers.Constant(0.1)
      x = keras.layers.Dense(4, name="layer0",
                             kernel_initializer=init)(input_layer)
      x = keras.layers.Dense(2, name="layer1", kernel_initializer=init)(x)
      m = ipu_keras.Model(input_layer, x, gradient_accumulation_count=3)

      m.compile('sgd', loss='mse')

      # Input data
      input_x = np.full([33, 32], 1.0, dtype=np.single)
      input_y = np.full([33, 1], 1.0, dtype=np.single)

      # Generate predictions.
      with self.assertRaisesRegex(
          ValueError, r"requires the number of steps in an epoch"
          r" 'steps_per_epoch' \(4\) to be evenly"
          r" divisible by the number of steps per execution of the"
          r" on-device training loop 'steps_per_run' \(3\)"
          r" multiplied by the replication factor \(2\)."):
        m.fit(input_x,
              input_y,
              batch_size=2,
              steps_per_epoch=4,
              steps_per_run=3)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.run_v2_only
  def testPredictWithNumpyDataInsufficientSamplesForParameters(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()

    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32),
                                       dtype=np.single,
                                       batch_size=2)
      init = keras.initializers.Constant(0.1)
      x = keras.layers.Dense(4, name="layer0",
                             kernel_initializer=init)(input_layer)
      x = keras.layers.Dense(2, name="layer1", kernel_initializer=init)(x)
      m = ipu_keras.Model(input_layer, x, gradient_accumulation_count=3)

      m.compile('sgd', loss='mse')

      # Input data
      input_x = np.full([33, 32], 1.0, dtype=np.single)

      # Generate predictions.
      with self.assertRaisesRegex(
          ValueError,
          r"The number of mini-batches in the dataset \(16\) must be"
          r" at least the gradient accumulation count \(3\)"
          r" multiplied by the replication factor \(2\) multiplied by"
          r" steps_per_run \(3\)."):
        m.predict(input_x, batch_size=2, steps_per_run=3)


if __name__ == "__main__":
  googletest.main()
