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
from absl.testing import parameterized

from tensorflow.python.ipu.config import IPUConfig
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import googletest
from tensorflow.python.ipu import ipu_strategy


def test_dataset(length=None, batch_size=1, x_val=1.0, y_val=0.2):
  constant_d = constant_op.constant(x_val, shape=[32])
  constant_l = constant_op.constant(y_val, shape=[2])

  ds = dataset_ops.Dataset.from_tensors((constant_d, constant_l))
  ds = ds.repeat(length)
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


def simple_sequential_model():
  return keras.Sequential([
      keras.layers.Flatten(),
      keras.layers.Dense(4),
      keras.layers.Dense(8),
  ])


def simple_functional_model():
  d = keras.layers.Input(32)
  x = keras.layers.Flatten()(d)
  x = keras.layers.Dense(4)(x)
  x = keras.layers.Dense(8)(x)
  return keras.Model(d, x)


class KerasModelExecutionParametersTest(test_util.TensorFlowTestCase,
                                        parameterized.TestCase):
  @tu.test_uses_ipus(num_ipus=8)
  @test_util.run_v2_only
  def testMismatchStepsPerExecutionAndReplicationFactor(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 8
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = simple_sequential_model()
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          RuntimeError,
          r"Currently `steps_per_execution` is set to 1 and the current IPU "
          r"system configuration and model configuration means that your Keras "
          r"model will automatically execute in a data-parallel fashion across "
          r"8 replicas"):
        m.fit(test_dataset(length=4))

  @tu.test_uses_ipus(num_ipus=8)
  @test_util.run_v2_only
  def testMismatchStepsPerExecutionAndReplicationFactorAndTruncated(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 8
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = simple_sequential_model()
      m.compile('sgd', loss='mse', steps_per_execution=8)

      with self.assertRaisesRegex(
          RuntimeError,
          r"Currently `steps_per_execution` is set to 7 \(truncated from 8 due "
          r"to 7 steps per epoch\) and the current IPU."):
        m.fit(test_dataset(length=7))

  @parameterized.parameters([simple_sequential_model, simple_functional_model])
  @tu.test_uses_ipus(num_ipus=1)
  @test_util.run_v2_only
  def testGradientAccumulation(self, model_fn):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = model_fn()
      m.compile('sgd', loss='mse', steps_per_execution=8)

      with self.assertRaisesRegex(
          RuntimeError,
          r"The model has been configured to use gradient accumulation for "
          r"training, however the current `steps_per_execution` value \(set to "
          r"8\) is not divisible by `gradient_accumulation_steps_per_replica "
          r"\* number of replicas` \(`gradient_accumulation_steps_per_replica` "
          r"is set to 3 and there are 1 replicas\)"):
        m.set_gradient_accumulation_options(
            gradient_accumulation_steps_per_replica=3)
        m.fit(test_dataset(length=8))

      with self.assertRaisesRegex(
          RuntimeError,
          r"The model has been configured to use gradient accumulation for "
          r"training, however the current `steps_per_execution` value \(set to "
          r"7 - truncated from 8 due to 7 steps per epoch\) is not divisible "
          r"by `gradient_accumulation_steps_per_replica \* number of replicas` "
          r"\(`gradient_accumulation_steps_per_replica` is set to 3 and there "
          r"are 1 replicas\)"):
        m.set_gradient_accumulation_options(
            gradient_accumulation_steps_per_replica=3)
        m.fit(test_dataset(length=7))

  @parameterized.parameters([simple_sequential_model, simple_functional_model])
  @tu.test_uses_ipus(num_ipus=8)
  @test_util.run_v2_only
  def testGradientAccumulationReplicated(self, model_fn):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 8
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = model_fn()
      m.compile('sgd', loss='mse', steps_per_execution=64)

      with self.assertRaisesRegex(
          RuntimeError,
          r"The model has been configured to use gradient accumulation for "
          r"training, however the current `steps_per_execution` value \(set to "
          r"64\) is not divisible by `gradient_accumulation_steps_per_replica "
          r"\* number of replicas` \(`gradient_accumulation_steps_per_replica` "
          r"is set to 3 and there are 8 replicas\)"):
        m.set_gradient_accumulation_options(
            gradient_accumulation_steps_per_replica=3)
        m.fit(test_dataset(length=64))


if __name__ == "__main__":
  googletest.main()
