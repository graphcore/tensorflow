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

import numpy as np

from tensorflow.python import ipu
from tensorflow.python.ipu import config
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu.keras import extensions
from tensorflow.python.eager import def_function
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training as training_module
from tensorflow.python.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras import optimizer_v2
from tensorflow.python.keras import preprocessing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


def get_imdb_dataset(num_words, maxlen):
  (x_train, y_train), (_, _) = imdb.load_data(num_words=num_words)
  x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)

  ds = dataset_ops.Dataset.from_tensor_slices((x_train, y_train))
  ds = ds.repeat()
  ds = ds.map(lambda x, y: (x, math_ops.cast(y, np.int32)))
  ds = ds.batch(32, drop_remainder=True)
  return ds


class KerasExtensionsTest(test.TestCase):
  @test_util.run_v2_only
  def testExtension(self):
    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    class TestLayer(base_layer.Layer):
      @base_layer.extension_delegate
      def call(self, x, training=False):  # pylint: disable=unused-argument,arguments-differ
        return x + 1.

    strategy = ipu_strategy.IPUStrategyV1()

    @def_function.function(jit_compile=True)
    def fn(layer):
      return layer(10.)

    def inner_test(expected_value, extension):
      strategy._register_keras_extension(TestLayer, extension)  # pylint: disable=protected-access
      with strategy.scope():
        l = TestLayer()
        result = strategy.run(fn, args=[l])
        self.assertAllClose(result, expected_value)
      strategy._delete_keras_extension(TestLayer)  # pylint: disable=protected-access

    # Test replacement.
    class TestExtension1(base_layer.KerasExtension):
      def _call_supported(self, *args, **kwargs):  # pylint: disable=unused-argument
        return True

      def _call_delegate(self, x, training=False):  # pylint: disable=unused-argument
        return x * x

    inner_test(100., TestExtension1)

    # Test replacement fails - not supported.
    class TestExtension2(base_layer.KerasExtension):
      def _call_supported(self, *args, **kwargs):  # pylint: disable=unused-argument
        return False

      def _call_delegate(self, x, training=False):  # pylint: disable=unused-argument
        return x * x

    inner_test(11., TestExtension2)

    # Test replacement fails - not implemented.
    class TestExtension3(base_layer.KerasExtension):
      def _call_supported(self, *args, **kwargs):  # pylint: disable=unused-argument
        return False

    inner_test(11., TestExtension3)

    # Test replacement fails - mismatch in spec.
    class TestExtension4(base_layer.KerasExtension):
      def _call_supported(self, *args, **kwargs):  # pylint: disable=unused-argument
        return False

      def _call_delegate(self, x, training=True):  # pylint: disable=unused-argument
        return x * x

    inner_test(11., TestExtension4)

    # Test replacement and calling through.
    class TestExtension5(base_layer.KerasExtension):
      def _call_supported(self, *args, **kwargs):  # pylint: disable=unused-argument
        return True

      def _call_delegate(self, x, training=False):  # pylint: disable=unused-argument
        return x * self.call(x, __extension_delegate=False)

    inner_test(110., TestExtension5)

  @test_util.run_v2_only
  def testExtensionWithClassMethod(self):
    # Ensure we can provide a delegate for a class method

    class TestLayer(base_layer.Layer):
      @classmethod
      @base_layer.extension_delegate
      def my_classmethod(cls, x, **kwargs):  # pylint: disable=unused-argument
        cls.method_was_called_times += 1
        self.assertIs(cls, TestLayer)
        return x * 123

    class TestLayer2(TestLayer):
      @classmethod
      def my_classmethod(cls, x, **kwargs):
        cls.method_was_called_times += 1
        self.assertIs(cls, TestLayer2)

        x = cls.__base__.my_classmethod(x)
        return x * 2

    class TestExtension(base_layer.KerasExtension):
      @classmethod
      def _my_classmethod_supported(cls, *args, **kwargs):  # pylint: disable=unused-argument
        return True

      @classmethod
      def _my_classmethod_delegate(cls, x, **kwargs):  # pylint: disable=unused-argument
        __class__.method_was_called_times += 1
        return x * cls.my_classmethod(x, __extension_delegate=False)

    def reset_spies():
      TestLayer.method_was_called_times = 0
      TestLayer2.method_was_called_times = 0
      TestExtension.method_was_called_times = 0

    strategy = ipu_strategy.IPUStrategyV1()
    strategy._register_keras_extension(TestLayer, TestExtension)  # pylint: disable=protected-access
    with strategy.scope():
      reset_spies()
      output = TestLayer.my_classmethod(2)
      self.assertEqual(TestLayer.method_was_called_times, 1)
      self.assertEqual(TestLayer2.method_was_called_times, 0)
      self.assertEqual(TestExtension.method_was_called_times, 1)
      self.assertEqual(output, 492)

      # Ensure we can provide a delegate even on a subclass.
      reset_spies()
      output = TestLayer2.my_classmethod(2)
      self.assertEqual(TestLayer.method_was_called_times, 1)
      self.assertEqual(TestLayer2.method_was_called_times, 1)
      self.assertEqual(TestExtension.method_was_called_times, 1)
      self.assertEqual(output, 984)

    # And we don't call the extension delegate outside the IPUStrategy.
    reset_spies()
    output = TestLayer.my_classmethod(1)
    self.assertEqual(TestLayer.method_was_called_times, 1)
    self.assertEqual(TestLayer2.method_was_called_times, 0)
    self.assertEqual(TestExtension.method_was_called_times, 0)
    self.assertEqual(output, 123)

    reset_spies()
    output = TestLayer2.my_classmethod(1)
    self.assertEqual(TestLayer.method_was_called_times, 1)
    self.assertEqual(TestLayer2.method_was_called_times, 1)
    self.assertEqual(TestExtension.method_was_called_times, 0)
    self.assertEqual(output, 246)

  @test_util.run_v2_only
  def testKerasModelExtensions(self):
    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      # Check sequential inheritance.
      model = sequential.Sequential()
      self.assertIsInstance(
          model, extensions.sequential_extensions.SequentialExtension)
      self.assertFalse(
          isinstance(model,
                     extensions.functional_extensions.FunctionalExtension))

      # Check models inhereting from Sequential are extended correctly.
      class MySequential(sequential.Sequential):
        pass

      model = MySequential()
      self.assertIsInstance(
          model, extensions.sequential_extensions.SequentialExtension)
      self.assertFalse(
          isinstance(model,
                     extensions.functional_extensions.FunctionalExtension))

      # Check functional Models are extended correctly.
      inp = layers.Input(shape=[])
      out = inp + inp
      model = training_module.Model(inp, out)
      self.assertIsInstance(
          model, extensions.functional_extensions.FunctionalExtension)

      # Check models inhereting from Model are extended correctly.
      class MyModel(training_module.Model):  # pylint: disable=abstract-method
        pass

      inputs = layers.Input(shape=(1,))
      outputs = layers.Dense(1)(inputs)
      model = MyModel(inputs, outputs)
      self.assertIsInstance(
          model, extensions.functional_extensions.FunctionalExtension)

  @test_util.run_v2_only
  def testModelOverrides(self):
    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():

      class MakeFunctionsOverrideModel(functional.Functional):
        def make_train_function(self, *args, **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
          return super().make_train_function(*args, **kwargs)

        def make_test_function(self, *args, **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
          return super().make_test_function(*args, **kwargs)

        def make_predict_function(self, *args, **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
          return super().make_predict_function(*args, **kwargs)

      inputs = layers.Input(shape=(1,))
      outputs = layers.Dense(1)(inputs)
      model = MakeFunctionsOverrideModel(inputs, outputs)
      model.compile(loss='mse')
      with self.assertRaisesRegex(RuntimeError,
                                  "The function `make_train_function`"):
        model.fit([1.], [1.], batch_size=1)
      with self.assertRaisesRegex(RuntimeError,
                                  "The function `make_test_function`"):
        model.evaluate([1.], [1.], batch_size=1)
      with self.assertRaisesRegex(RuntimeError,
                                  "The function `make_predict_function`"):
        model.predict([1.], batch_size=1)

      class StepFunctionsOverrideModel(functional.Functional):
        def train_step(self, *args, **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
          return super().train_step(*args, **kwargs)

        def test_step(self, *args, **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
          return super().test_step(*args, **kwargs)

        def predict_step(self, *args, **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
          return super().predict_step(*args, **kwargs)

      inputs = layers.Input(shape=(1,))
      outputs = layers.Dense(1)(inputs)
      model = StepFunctionsOverrideModel(inputs, outputs)
      model.compile(loss='mse')
      assignments = model.get_pipeline_stage_assignment()
      for assignment in assignments:
        assignment.pipeline_stage = 0
      model.set_pipeline_stage_assignment(assignments)
      model.set_pipelining_options(gradient_accumulation_steps_per_replica=1)
      with self.assertRaisesRegex(RuntimeError, "The function `train_step`"):
        model.fit([1.], [1.], batch_size=1)
      with self.assertRaisesRegex(RuntimeError, "The function `test_step`"):
        model.evaluate([1.], [1.], batch_size=1)
      with self.assertRaisesRegex(RuntimeError, "The function `predict_step`"):
        model.predict([1.], batch_size=1)
      model.reset_pipeline_stage_assignment()
      model.compile(steps_per_execution=2)
      model.set_gradient_accumulation_options(
          gradient_accumulation_steps_per_replica=2)
      with self.assertRaisesRegex(RuntimeError, "The function `train_step`"):
        model.fit([[1.], [1.]], [[1.], [1.]], batch_size=1)

      class CallOverrideModel(functional.Functional):
        def call(self, *args, **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
          return super().call(*args, **kwargs)

      inputs = layers.Input(shape=(1,))
      outputs = layers.Dense(1)(inputs)
      model = CallOverrideModel(inputs, outputs)
      model.compile(loss='mse')
      assignments = model.get_pipeline_stage_assignment()
      for assignment in assignments:
        assignment.pipeline_stage = 0
      model.set_pipeline_stage_assignment(assignments)
      model.set_pipelining_options(gradient_accumulation_steps_per_replica=1)
      with self.assertRaisesRegex(RuntimeError, "The function `call`"):
        model.fit([1.], [1.], batch_size=1)
      with self.assertRaisesRegex(RuntimeError, "The function `call`"):
        model.evaluate([1.], [1.], batch_size=1)
      with self.assertRaisesRegex(RuntimeError, "The function `call`"):
        model.predict([1.], batch_size=1)

  @test_util.run_v2_only
  def testSequential(self):
    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():

      class MakeFunctionsOverrideModel(sequential.Sequential):
        def make_train_function(self, *args, **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
          return super().make_train_function(*args, **kwargs)

        def make_test_function(self, *args, **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
          return super().make_test_function(*args, **kwargs)

        def make_predict_function(self, *args, **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
          return super().make_predict_function(*args, **kwargs)

      model = MakeFunctionsOverrideModel([layers.Dense(1)])
      model.compile(loss='mse')
      with self.assertRaisesRegex(RuntimeError,
                                  "The function `make_train_function`"):
        model.fit([1.], [1.], batch_size=1)
      with self.assertRaisesRegex(RuntimeError,
                                  "The function `make_test_function`"):
        model.evaluate([1.], [1.], batch_size=1)
      with self.assertRaisesRegex(RuntimeError,
                                  "The function `make_predict_function`"):
        model.predict([1.], batch_size=1)

      class StepFunctionsOverrideModel(sequential.Sequential):
        def train_step(self, *args, **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
          return super().train_step(*args, **kwargs)

        def test_step(self, *args, **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
          return super().test_step(*args, **kwargs)

        def predict_step(self, *args, **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
          return super().predict_step(*args, **kwargs)

      model = StepFunctionsOverrideModel([layers.Dense(1)])
      model.compile(loss='mse')
      model.set_pipeline_stage_assignment([0])
      model.set_pipelining_options(gradient_accumulation_steps_per_replica=1)
      with self.assertRaisesRegex(RuntimeError, "The function `train_step`"):
        model.fit([1.], [1.], batch_size=1)
      with self.assertRaisesRegex(RuntimeError, "The function `test_step`"):
        model.evaluate([1.], [1.], batch_size=1)
      with self.assertRaisesRegex(RuntimeError, "The function `predict_step`"):
        model.predict([1.], batch_size=1)
      model.reset_pipeline_stage_assignment()
      model.compile(steps_per_execution=2)
      model.set_gradient_accumulation_options(
          gradient_accumulation_steps_per_replica=2)
      with self.assertRaisesRegex(RuntimeError, "The function `train_step`"):
        model.fit([[1.], [1.]], [[1.], [1.]], batch_size=1)

      class CallOverrideModel(sequential.Sequential):
        def call(self, *args, **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
          return super().call(*args, **kwargs)

      model = CallOverrideModel([layers.Dense(1)])
      model.compile(loss='mse')
      model.set_pipeline_stage_assignment([0])
      model.set_pipelining_options(gradient_accumulation_steps_per_replica=1)
      with self.assertRaisesRegex(RuntimeError, "The function `call`"):
        model.fit([1.], [1.], batch_size=1)
      with self.assertRaisesRegex(RuntimeError, "The function `call`"):
        model.evaluate([1.], [1.], batch_size=1)
      with self.assertRaisesRegex(RuntimeError, "The function `call`"):
        model.predict([1.], batch_size=1)

  @test_util.run_v2_only
  def testModel(self):
    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():

      class MakeFunctionsOverrideModel(training_module.Model):  # pylint: disable=abstract-method
        def __init__(self, dense_layers):
          super(MakeFunctionsOverrideModel, self).__init__()
          self.dense_layers = dense_layers

        def make_train_function(self, *args, **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
          return super().make_train_function(*args, **kwargs)

        def make_test_function(self, *args, **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
          return super().make_test_function(*args, **kwargs)

        def make_predict_function(self, *args, **kwargs):  # pylint: disable=useless-super-delegation,arguments-differ
          return super().make_predict_function(*args, **kwargs)

        def call(self, inputs):  # pylint: disable=arguments-differ
          x = inputs
          for layer in self.dense_layers:
            x = layer(x)
          return x

      model = MakeFunctionsOverrideModel([layers.Dense(1)])
      model.compile(loss='mse')
      with self.assertRaisesRegex(RuntimeError,
                                  "The function `make_train_function`"):
        model.fit([[1.]], [[1.]], batch_size=1)
      with self.assertRaisesRegex(RuntimeError,
                                  "The function `make_test_function`"):
        model.evaluate([[1.]], [[1.]], batch_size=1)
      with self.assertRaisesRegex(RuntimeError,
                                  "The function `make_predict_function`"):
        model.predict([[1.]], batch_size=1)

      class CallNotImplementedModel(training_module.Model):  # pylint: disable=abstract-method
        pass

      model = CallNotImplementedModel()
      model.compile(loss='mse')
      with self.assertRaisesRegex(RuntimeError, "The function `call`"):
        model.fit([[1.]], [[1.]], batch_size=1)
      with self.assertRaisesRegex(RuntimeError, "The function `call`"):
        model.evaluate([[1.]], [[1.]], batch_size=1)
      with self.assertRaisesRegex(RuntimeError, "The function `call`"):
        model.predict([[1.]], batch_size=1)

  @test_util.run_v2_only
  def testInputValidation(self):
    class TwoInTwoOutIdentity(layers.Layer):
      def __init__(self, **layer_args):
        super(TwoInTwoOutIdentity, self).__init__(**layer_args)

      def call(self, x, y):  # pylint: disable=arguments-differ
        return x, y

    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_x = layers.Input(shape=(80), dtype=np.int32, batch_size=32)
      input_y = layers.Input(shape=(1,), dtype=np.int32, batch_size=32)

      with ipu.keras.PipelineStage(0):
        x = layers.Dense(10)(input_x)
      with ipu.keras.PipelineStage(1):
        x = layers.Dense(1)(x)
        x = TwoInTwoOutIdentity()(x, input_y)[0]

      # Setup a model that requires 2 inputs but is given a dataset that
      # only generates one, which should fail with an exception saying as much.
      model = Model(inputs=[input_x, input_y], outputs=x)
      model.set_pipelining_options(gradient_accumulation_steps_per_replica=8,
                                   device_mapping=[0, 1])
      model.compile(loss='binary_crossentropy',
                    optimizer=optimizer_v2.adam.Adam(0.005),
                    steps_per_execution=16)

      dataset = get_imdb_dataset(num_words=2000, maxlen=80)
      expected_message_regex = r"ValueError: Layer \w+ expects 2 input\(s\)," +\
      r" but it received 1 input tensors\. Inputs received: .*"
      self.assertRaisesRegex(ValueError,
                             expected_message_regex,
                             model.fit,
                             dataset,
                             steps_per_epoch=768,
                             epochs=3)


if __name__ == '__main__':
  test.main()
