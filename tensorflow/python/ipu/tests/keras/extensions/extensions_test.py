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

from tensorflow.python.ipu import config
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu.keras import extensions
from tensorflow.python.eager import def_function
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training as training_module
from tensorflow.python.keras import layers
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


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

    @def_function.function(experimental_compile=True)
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


if __name__ == '__main__':
  test.main()
