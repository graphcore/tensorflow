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
from tensorflow.python.platform import googletest
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import keras as ipu_keras


def dataset_fn(batch_size=1):
  features = [i * np.ones((4), dtype=np.float32) for i in range(10)]
  labels = [i * np.ones(1, dtype=np.int32) for i in range(10)]
  ds = dataset_ops.Dataset.from_tensor_slices((features, labels))
  ds = ds.repeat()
  return ds.batch(batch_size, drop_remainder=True)


def predict_input_fn(length):
  inputs = [(i % 10) * np.ones((4, 4), dtype=np.float32)
            for i in range(length)]
  ds = dataset_ops.Dataset.from_tensor_slices((inputs))
  return ds


class KerasModelsTests(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=4)
  @test_util.run_v2_only
  def test_sequential_and_pipeline(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    def stage1():
      return [
          keras.layers.Dense(
              4,
              activation="relu",
              kernel_initializer=keras.initializers.Constant(0.5))
      ]

    def stage2():
      return [
          keras.layers.Dense(
              4, kernel_initializer=keras.initializers.Constant(0.5))
      ]

    m = keras.Sequential(stage1() + stage2())
    m.compile(optimizer="adam", loss='mae')

    # Fit the weights to the dataset
    m.fit(dataset_fn(16), steps_per_epoch=1, epochs=2)
    predict_cpu = m.predict(predict_input_fn(32), steps=8)
    cpu_weights = m.weights

    strategy_seq = ipu_strategy.IPUStrategyV1()
    with strategy_seq.scope():
      m = keras.Sequential(stage1() + stage2())
      m.set_gradient_accumulation_options(
          gradient_accumulation_steps_per_replica=4,
          experimental_normalize_gradients=True)
      m.compile(optimizer="adam", loss='mae', steps_per_execution=16)

      # Fit the weights to the dataset
      m.fit(dataset_fn(), steps_per_epoch=16, epochs=2)
      ipu_weights = m.weights

      # Generate predictions
      predict_seq = m.predict(predict_input_fn(32), steps=8)

    self.assertAllClose(cpu_weights, ipu_weights)

    strategy_pipeline = ipu_strategy.IPUStrategyV1()
    with strategy_pipeline.scope():
      m = keras.Sequential(stage1() + stage2())
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=8,
                               experimental_normalize_gradients=True)
      m.set_pipeline_stage_assignment([0, 1])
      m.compile(optimizer="adam", loss='mae', steps_per_execution=16)

      # Fit the weights to the dataset
      m.fit(dataset_fn(), steps_per_epoch=16, epochs=2)
      ipu_weights = m.weights

      # Generate predictions
      predict_pipeline = m.predict(predict_input_fn(32), steps=8)

    self.assertAllClose(cpu_weights, ipu_weights)
    self.assertAllClose(predict_seq, predict_pipeline)
    self.assertAllClose(predict_seq, predict_cpu)

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.run_v2_only
  def test_functional_and_pipeline(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    def get_model(n=3, pipeline_stages=None):
      input_layer = keras.layers.Input(shape=(4))
      x = input_layer
      if pipeline_stages:
        assert len(pipeline_stages) == n
        for s in pipeline_stages:
          with ipu_keras.PipelineStage(s):
            x = keras.layers.Dense(
                4,
                activation=keras.activations.relu,
                kernel_initializer=keras.initializers.Constant(0.5))(x)
      else:
        for _ in range(n):
          x = keras.layers.Dense(
              4,
              activation=keras.activations.relu,
              kernel_initializer=keras.initializers.Constant(0.5))(x)

      return input_layer, x

    strategy_model = ipu_strategy.IPUStrategyV1()
    with strategy_model.scope():
      m = keras.Model(*get_model())
      m.set_gradient_accumulation_options(
          gradient_accumulation_steps_per_replica=6,
          experimental_normalize_gradients=True)
      m.compile(optimizer="adam", loss='mae', steps_per_execution=24)

      # Fit the weights to the dataset
      m.fit(dataset_fn(), steps_per_epoch=24, epochs=2)
      model_weights = m.weights

      # Generate predictions
      predict_model = m.predict(predict_input_fn(48))

    strategy_pipeline = ipu_strategy.IPUStrategyV1()
    with strategy_pipeline.scope():
      m = keras.Model(*get_model(pipeline_stages=[0, 1, 2]))
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=12,
                               experimental_normalize_gradients=True,
                               device_mapping=[1, 0, 1])
      m.compile(optimizer="adam", loss='mae', steps_per_execution=24)

      # Fit the weights to the dataset
      m.fit(dataset_fn(), steps_per_epoch=24, epochs=2)
      pipeline_weights = m.weights

      # Generate predictions
      predict_pipeline = m.predict(predict_input_fn(48))

    self.assertAllClose(pipeline_weights, model_weights)
    self.assertAllClose(predict_model, predict_pipeline)

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.run_v2_only
  def test_functional_and_sequential(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    dataset_length = 32

    def get_sequential_model():
      return [
          keras.layers.Dense(
              4,
              activation=keras.activations.relu,
              kernel_initializer=keras.initializers.Constant(0.5),
              name="1"),
          keras.layers.Dense(
              4,
              activation=keras.activations.relu,
              kernel_initializer=keras.initializers.Constant(0.45),
              name="2"),
          keras.layers.Dense(
              4,
              activation=keras.activations.relu,
              kernel_initializer=keras.initializers.Constant(0.55),
              name="3"),
          keras.layers.Dense(
              4,
              kernel_initializer=keras.initializers.Constant(0.61),
              name="4")
      ]

    def get_model():
      input_layer = keras.layers.Input(shape=(4))
      output = input_layer
      for l in get_sequential_model():
        output = l(output)
      return keras.Model(input_layer, output)

    strategy_seq = ipu_strategy.IPUStrategyV1()
    with strategy_seq.scope():
      m = keras.Sequential(get_sequential_model())
      m.compile("adam", loss='mse', steps_per_execution=4)

      # Fit the weights to the dataset
      history_seq = m.fit(dataset_fn(), steps_per_epoch=8, epochs=2)

      # Generate predictions
      predict_seq = m.predict(predict_input_fn(48))

    strategy_model = ipu_strategy.IPUStrategyV1()
    with strategy_model.scope():
      m = get_model()
      m.compile("adam", loss='mse', steps_per_execution=4)

      # Fit the weights to the dataset
      history_model = m.fit(dataset_fn(), steps_per_epoch=8, epochs=2)

      # Generate predictions
      predict_model = m.predict(predict_input_fn(48))

    self.assertAllClose(history_seq.history, history_model.history)
    self.assertAllClose(predict_model, predict_seq)


if __name__ == "__main__":
  googletest.main()
