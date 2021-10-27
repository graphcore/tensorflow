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
"""Test for IPU Keras single IPU model."""

import os
from tensorflow.python.ipu.config import IPUConfig
import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


def dataset_with_labels():
  x1 = np.ones((8), dtype=np.float64)
  x2 = np.ones((8), dtype=np.float64)
  x3 = np.ones((8), dtype=np.float64)
  y1 = np.ones((1), dtype=np.float64)
  y2 = np.ones((1), dtype=np.float64)
  ds_x = dataset_ops.Dataset.from_tensors((x1, x2, x3))
  ds_y = dataset_ops.Dataset.from_tensors((y1, y2))
  ds_xy = dataset_ops.Dataset.zip(
      (ds_x, ds_y)).repeat(16).batch(2, drop_remainder=True)
  return ds_xy


def numpy_data():
  x1 = np.ones((2, 8), dtype=np.float64)
  x2 = np.ones((2, 8), dtype=np.float64)
  x3 = np.ones((2, 8), dtype=np.float64)
  return (x1, x2, x3)


def model_fn(nested_outputs=False):
  input_1 = keras.Input(8)
  input_2 = keras.Input(8)
  input_3 = keras.Input(8)

  init = keras.initializers.Constant(1)

  cat = keras.layers.Concatenate()([input_1, input_2, input_3])

  dense_3 = keras.layers.Dense(1,
                               kernel_initializer=init,
                               activation=keras.activations.relu,
                               name="output1")(cat)
  dense_4 = keras.layers.Dense(2,
                               kernel_initializer=init,
                               activation=keras.activations.relu,
                               name="output2")(cat)

  if nested_outputs:
    return ((input_1, input_2, input_3), ((dense_3,), (dense_4,)))

  return ((input_1, input_2, input_3), (dense_3, dense_4))


class KerasSyntheticDataTest(test.TestCase):
  @tu.skip_on_hw
  @test_util.run_v2_only
  def testSyntheticDataFit(self):
    poplar_flags = os.environ.get("TF_POPLAR_FLAGS", "")
    poplar_flags += " --use_synthetic_data"

    with test.mock.patch.dict("os.environ", {"TF_POPLAR_FLAGS": poplar_flags}):
      strategy = ipu.ipu_strategy.IPUStrategyV1()
      with strategy.scope():
        cfg = IPUConfig()
        cfg.auto_select_ipus = 1
        cfg.configure_ipu_system()

        model = keras.Model(*model_fn())
        model.compile('sgd', ['mse', 'mse'], metrics=['accuracy'])

        model.fit(dataset_with_labels(), epochs=4)

  @tu.skip_on_hw
  @test_util.run_v2_only
  def testSyntheticDataEvaluate(self):
    poplar_flags = os.environ.get("TF_POPLAR_FLAGS", "")
    poplar_flags += " --use_synthetic_data"

    with test.mock.patch.dict("os.environ", {"TF_POPLAR_FLAGS": poplar_flags}):
      strategy = ipu.ipu_strategy.IPUStrategyV1()
      with strategy.scope():
        cfg = IPUConfig()
        cfg.auto_select_ipus = 1
        cfg.configure_ipu_system()

        model = keras.Model(*model_fn())
        model.compile('sgd', ['mse', 'mse'], metrics=['accuracy'])

        model.evaluate(dataset_with_labels())

  @tu.skip_on_hw
  @test_util.run_v2_only
  def testSyntheticDataPredict(self):
    poplar_flags = os.environ.get("TF_POPLAR_FLAGS", "")
    poplar_flags += " --use_synthetic_data"

    with test.mock.patch.dict("os.environ", {"TF_POPLAR_FLAGS": poplar_flags}):
      strategy = ipu.ipu_strategy.IPUStrategyV1()
      with strategy.scope():
        cfg = IPUConfig()
        cfg.auto_select_ipus = 1
        cfg.configure_ipu_system()

        model = keras.Model(*model_fn())
        model.compile('sgd', ['mse', 'mse'], metrics=['accuracy'])

        model.predict(numpy_data(), batch_size=2)

  @tu.skip_on_hw
  @test_util.run_v2_only
  def testSyntheticDataPredictNestedOutput(self):
    poplar_flags = os.environ.get("TF_POPLAR_FLAGS", "")
    poplar_flags += " --use_synthetic_data"

    with test.mock.patch.dict("os.environ", {"TF_POPLAR_FLAGS": poplar_flags}):
      strategy = ipu.ipu_strategy.IPUStrategy()
      with strategy.scope():
        cfg = IPUConfig()
        tu.enable_ipu_events(cfg)
        cfg.auto_select_ipus = 1
        cfg.configure_ipu_system()

        model = keras.Model(*model_fn(nested_outputs=True))
        model.compile('sgd', ['mse', 'mse'], metrics=['accuracy'])

        model.predict(numpy_data(), batch_size=2)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.run_v2_only
  def testSyntheticDataWithReplication(self):
    poplar_flags = os.environ.get("TF_POPLAR_FLAGS", "")
    poplar_flags += " --use_synthetic_data"

    with test.mock.patch.dict("os.environ", {"TF_POPLAR_FLAGS": poplar_flags}):
      strategy = ipu.ipu_strategy.IPUStrategyV1()
      with strategy.scope():
        cfg = IPUConfig()
        cfg.auto_select_ipus = 2
        tu.add_hw_ci_connection_options(cfg)
        cfg.configure_ipu_system()

        model = keras.Model(*model_fn())
        model.compile('sgd', ['mse', 'mse'],
                      metrics=['accuracy'],
                      steps_per_execution=8)

        model.fit(dataset_with_labels(), epochs=4)


if __name__ == '__main__':
  test.main()
