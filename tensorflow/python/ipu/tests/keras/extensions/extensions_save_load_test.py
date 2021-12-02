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
"""Test for KerasExtension interface."""
import tempfile
import numpy as np
from tensorflow.python import ipu

from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu.keras import extensions
from tensorflow.python.keras import models
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import training as training_module
from tensorflow.python.keras import layers
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.data.ops import dataset_ops

IN_SHAPE = (10, 32)
OUT_CHANNELS = 16
NUM_SAMPLES = 4
BATCH_SIZE = 2


class KerasExtensionsSaveLoadTest(test.TestCase):
  @test_util.run_v2_only
  def testFunctionalExtension(self):
    strategy = ipu_strategy.IPUStrategyV1()

    class TestExtension(extensions.functional_extensions.FunctionalExtension):  # pylint: disable=abstract-method
      def __init__(self):
        extensions.functional_extensions.FunctionalExtension.__init__(self)
        self.test_option = False

      @property
      def test_option(self):
        return self._test_option

      @test_option.setter
      def test_option(self, value):
        self._test_option = value

      def _get_config_supported(self):
        return True

      def _get_config_delegate(self):
        config = self.get_config(__extension_delegate=False)
        config["test_option"] = self.test_option
        return config

      def _deserialize_from_config_supported(self, config):
        del config
        return True

      def _deserialize_from_config_delegate(self, config):
        self.test_option = config.get("test_option", self.test_option)

    with strategy.scope():
      # Replace the extension with the test version.
      strategy._register_keras_extension(functional.Functional, TestExtension)  # pylint: disable=protected-access

      inputs = {
          'x1': layers.Input(shape=(10,)),
          'x2': layers.Input(shape=(1,))
      }
      t = layers.Dense(1, activation='relu')(inputs['x1'])
      outputs = layers.Add()([t, inputs['x2']])
      model = functional.Functional(inputs, outputs)

      self.assertFalse(model.test_option)
      model.test_option = True

      # Test get_config/from_config.
      model_config = model.get_config()
      self.assertTrue(model_config["test_option"])
      loaded_model = functional.Functional.from_config(model_config)
      self.assertTrue(model.test_option)
      loaded_model = training_module.Model.from_config(model_config)
      self.assertTrue(model.test_option)

      # Test save/load (including SavedModel).
      with tempfile.TemporaryDirectory() as temp_dir:
        model.save(temp_dir)
        loaded_model = save.load_model(temp_dir)
        self.assertTrue(loaded_model.test_option)

      # Test json.
      loaded_model = models.model_from_json(model.to_json())
      self.assertTrue(model.test_option)

  @test_util.run_v2_only
  def testSequentialExtension(self):
    strategy = ipu_strategy.IPUStrategyV1()

    class TestExtension(extensions.sequential_extensions.SequentialExtension):  # pylint: disable=abstract-method
      def __init__(self):
        extensions.sequential_extensions.SequentialExtension.__init__(self)
        self.test_option = False

      @property
      def test_option(self):
        return self._test_option

      @test_option.setter
      def test_option(self, value):
        self._test_option = value

      def _get_config_supported(self):
        return True

      def _get_config_delegate(self):
        config = self.get_config(__extension_delegate=False)
        config["test_option"] = self.test_option
        return config

      def _deserialize_from_config_supported(self, config):
        del config
        return True

      def _deserialize_from_config_delegate(self, config):
        self.test_option = config.get("test_option", self.test_option)

    with strategy.scope():
      # Replace the extension with the test version.
      strategy._register_keras_extension(sequential.Sequential, TestExtension)  # pylint: disable=protected-access

      model = sequential.Sequential([layers.Dense(1, activation='relu')])
      self.assertFalse(model.test_option)
      model.test_option = True
      # Set the shape so that the model can be saved.
      model.build([None, 10])

      # Test get_config/from_config.
      model_config = model.get_config()
      self.assertTrue(model_config["test_option"])
      loaded_model = sequential.Sequential.from_config(model_config)
      self.assertTrue(model.test_option)

      # Test save/load (including SavedModel).
      with tempfile.TemporaryDirectory() as temp_dir:
        model.save(temp_dir)
        loaded_model = save.load_model(temp_dir)
        self.assertTrue(loaded_model.test_option)

      # Test json.
      loaded_model = models.model_from_json(model.to_json())
      self.assertTrue(model.test_option)

  @test_util.run_v2_only
  def testModelExtension(self):
    # Model subclasses do not support get_config()/from_config() by default,
    # unless they subclass Functional.

    np.random.seed(42)
    strategy = ipu_strategy.IPUStrategyV1()

    # We have to run the model first to be able to save it.
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.ipu_model.compile_ipu_code = False
    cfg.configure_ipu_system()

    def create_dataset(shape, num_samples, batch_size):
      data = np.random.uniform(size=(num_samples,) + shape).astype(np.float32)
      return dataset_ops.Dataset.from_tensor_slices(data).batch(
          batch_size, drop_remainder=True)

    class TestModel(training_module.Model):  # pylint: disable=abstract-method
      def __init__(self, in_channels=32, out_channels=16):
        super(TestModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dense_layers = [
            layers.Dense(in_channels, activation="relu"),
            layers.Dense(out_channels, activation="softmax")
        ]

      def call(self, inputs):  # pylint: disable=arguments-differ
        x = inputs
        for layer in self.dense_layers:
          x = layer(x)
        return x

    with strategy.scope():
      # Replace the extension with the test version.
      # pylint: disable=protected-access
      strategy._register_keras_extension(
          training_module.Model, extensions.model_extensions.ModelExtension)

      dataset = create_dataset(IN_SHAPE, NUM_SAMPLES, BATCH_SIZE)

      model = TestModel()
      self.assertIsInstance(model, training_module.Model)
      self.assertIsInstance(model, extensions.model_extensions.ModelExtension)
      model.predict(dataset)  # Compile and run the model.

      # We shouldn't be able to save this model's config, because it is not
      # defined in this example.
      self.assertFalse(model._get_config_supported())  # pylint: disable=protected-access
      self.assertRaises(NotImplementedError, model.get_config)
      self.assertRaises(NotImplementedError, model.to_json)

      # Test save/load (including SavedModel).
      with tempfile.TemporaryDirectory() as temp_dir:
        model.save(temp_dir)
        loaded_model = save.load_model(temp_dir,
                                       custom_objects={"TestModel": TestModel})
        loaded_model.compile()


if __name__ == '__main__':
  test.main()
