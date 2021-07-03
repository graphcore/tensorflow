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
from tensorflow.python.ipu import ipu_strategy


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
  @tu.test_uses_ipus(num_ipus=8)
  @test_util.run_v2_only
  def testMismatchStepsPerExecutionAndReplicationFactor(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 8
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(simple_model())
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
      m = keras.Sequential(simple_model())
      m.compile('sgd', loss='mse', steps_per_execution=8)

      with self.assertRaisesRegex(
          RuntimeError,
          r"Currently `steps_per_execution` is set to 7 \(truncated from 8 due "
          r"to 7 steps per epoch\) and the current IPU."):
        m.fit(test_dataset(length=7))


if __name__ == "__main__":
  googletest.main()
