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


def dataset_fn():
  features = [i * np.ones((4, 4), dtype=np.float32) for i in range(10)]
  labels = [i * np.ones(1, dtype=np.int32) for i in range(10)]
  ds = dataset_ops.Dataset.from_tensor_slices((features, labels))
  return ds.repeat()


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

    strategy_seq = ipu_strategy.IPUStrategy()
    with strategy_seq.scope():
      m = ipu_keras.Sequential(stage1() + stage2(),
                               gradient_accumulation_count=8)
      m.compile(optimizer="adam", loss='mae')

      # Fit the weights to the dataset
      history_seq = m.fit(dataset_fn(), steps_per_epoch=4, epochs=2)

      # Generate predictions
      predict_seq = m.predict(predict_input_fn(32), steps=4)

    strategy_pipeline = ipu_strategy.IPUStrategy()
    with strategy_pipeline.scope():
      m = ipu_keras.PipelineSequential([stage1(), stage2()],
                                       gradient_accumulation_count=16,
                                       device_mapping=[1, 0])
      m.compile(optimizer="adam", loss='mae')

      # Fit the weights to the dataset
      history_pipeline = m.fit(dataset_fn(), steps_per_epoch=2, epochs=2)

      # Generate predictions
      predict_pipeline = m.predict(predict_input_fn(32), steps=2)

    seq_loss = history_seq.history['loss']
    pipeline_loss = history_pipeline.history['loss']
    for s, p in zip(seq_loss, pipeline_loss):
      np.testing.assert_allclose(s, p)

    self.assertEqual(predict_seq.shape, predict_pipeline.shape)

    for s, p in zip(predict_seq, predict_pipeline):
      np.testing.assert_allclose(s, p)

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

    strategy_model = ipu_strategy.IPUStrategy()
    with strategy_model.scope():
      m = ipu_keras.Model(*get_model(), gradient_accumulation_count=12)
      m.compile(optimizer="adam", loss='mae')

      # Fit the weights to the dataset
      history_model = m.fit(dataset_fn(), steps_per_epoch=4, epochs=2)

      # Generate predictions
      predict_model = m.predict(predict_input_fn(48))

    strategy_pipeline = ipu_strategy.IPUStrategy()
    with strategy_pipeline.scope():
      m = ipu_keras.PipelineModel(*get_model(pipeline_stages=[0, 1, 2]),
                                  gradient_accumulation_count=24,
                                  device_mapping=[1, 0, 1])
      m.compile(optimizer="adam", loss='mae')

      # Fit the weights to the dataset
      history_pipeline = m.fit(dataset_fn(), steps_per_epoch=2, epochs=2)

      # Generate predictions
      predict_pipeline = m.predict(predict_input_fn(48))

    model_loss = history_model.history['loss']
    pipeline_loss = history_pipeline.history['loss']
    for s, p in zip(model_loss, pipeline_loss):
      np.testing.assert_allclose(s, p)

    self.assertEqual(predict_model.shape, predict_pipeline.shape)

    for s, p in zip(predict_model, predict_pipeline):
      np.testing.assert_allclose(s, p)

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.run_v2_only
  def test_functional_and_sequential(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    # Multiple of replication_factor (4) * gradient_accumulation_count (8)
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
      return ipu_keras.Model(input_layer,
                             output,
                             gradient_accumulation_count=8)

    strategy_seq = ipu_strategy.IPUStrategy()
    with strategy_seq.scope():
      m = ipu_keras.Sequential(get_sequential_model(),
                               gradient_accumulation_count=8)
      m.compile("adam", loss='mse')

      # Fit the weights to the dataset
      history_seq = m.fit(dataset_fn(), steps_per_epoch=4, epochs=2)

      # Generate predictions
      predict_seq = m.predict(predict_input_fn(dataset_length))

    strategy_model = ipu_strategy.IPUStrategy()
    with strategy_model.scope():
      m = get_model()
      m.compile("adam", loss='mse')

      # Fit the weights to the dataset
      history_model = m.fit(dataset_fn(), steps_per_epoch=4, epochs=2)

      # Generate predictions
      predict_model = m.predict(predict_input_fn(dataset_length))

    seq_loss = history_seq.history['loss']
    model_loss = history_model.history['loss']
    for l_seq, l_model in zip(seq_loss, model_loss):
      np.testing.assert_allclose(l_seq, l_model)

    self.assertEqual(predict_seq.shape, predict_model.shape)

    for t_seq, t_model in zip(predict_seq, predict_model):
      np.testing.assert_allclose(t_seq, t_model)


if __name__ == "__main__":
  googletest.main()
