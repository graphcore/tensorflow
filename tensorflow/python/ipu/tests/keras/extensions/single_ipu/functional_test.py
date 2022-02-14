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
#,============================================================================
"""Tests for layer graphs construction & handling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import input_layer as input_layer_lib
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python import ipu

try:
  import yaml  # pylint:disable=g-import-not-at-top
except ImportError:
  yaml = None


class NetworkConstructionTest(keras_parameterized.TestCase):
  def setUp(self):
    super(NetworkConstructionTest, self).setUp()
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 1
    cfg.configure_ipu_system()
    self._ipu_strategy = ipu.ipu_strategy.IPUStrategyV1()
    self._ipu_strategy_scope = self._ipu_strategy.scope()
    self._ipu_strategy_scope.__enter__()

  def tearDown(self):
    self._ipu_strategy_scope.__exit__(None, None, None)
    super(NetworkConstructionTest, self).tearDown()

  def test_activity_regularization_with_model_composition(self):
    def reg(x):
      return math_ops.reduce_sum(x)

    net_a_input = input_layer_lib.Input((2,))
    net_a = net_a_input
    net_a = layers.Dense(2,
                         kernel_initializer='ones',
                         use_bias=False,
                         activity_regularizer=reg)(net_a)
    model_a = training_lib.Model([net_a_input], [net_a])

    net_b_input = input_layer_lib.Input((2,))
    net_b = model_a(net_b_input)
    model_b = training_lib.Model([net_b_input], [net_b])

    model_b.compile(optimizer='sgd', loss=None)
    x = np.ones((1, 2))
    loss = model_b.evaluate(x, batch_size=1)
    self.assertEqual(loss, 4.)

  def test_layer_sharing_at_heterogenous_depth(self):
    x_val = np.random.random((10, 5))

    x = input_layer_lib.Input(shape=(5,))
    a = layers.Dense(5, name='A')
    b = layers.Dense(5, name='B')
    output = a(b(a(b(x))))
    m = training_lib.Model(x, output)

    output_val = m.predict(x_val, batch_size=10)

    config = m.get_config()
    weights = m.get_weights()

    m2 = models.Model.from_config(config)
    m2.set_weights(weights)

    output_val_2 = m2.predict(x_val, batch_size=10)
    self.assertAllClose(output_val, output_val_2, atol=1e-6)

  def test_layer_sharing_at_heterogenous_depth_with_concat(self):
    input_shape = (16, 9, 3)
    input_layer = input_layer_lib.Input(shape=input_shape)

    a = layers.Dense(3, name='dense_A')
    b = layers.Dense(3, name='dense_B')
    c = layers.Dense(3, name='dense_C')

    x1 = b(a(input_layer))
    x2 = a(c(input_layer))
    output = layers.concatenate([x1, x2])

    m = training_lib.Model(inputs=input_layer, outputs=output)

    x_val = np.random.random((10, 16, 9, 3))
    output_val = m.predict(x_val, batch_size=10)

    config = m.get_config()
    weights = m.get_weights()

    m2 = models.Model.from_config(config)
    m2.set_weights(weights)

    output_val_2 = m2.predict(x_val, batch_size=10)
    self.assertAllClose(output_val, output_val_2, atol=1e-6)

  def test_explicit_training_argument(self):
    a = layers.Input(shape=(2,))
    b = layers.Dropout(0.5)(a)
    base_model = training_lib.Model(a, b)

    a = layers.Input(shape=(2,))
    b = base_model(a, training=False)
    model = training_lib.Model(a, b)

    x = np.ones((100, 2))
    y = np.ones((100, 2))
    model.compile(optimizer='sgd', loss='mse')
    loss = model.train_on_batch(x, y)
    self.assertEqual(loss, 0)  # In inference mode, output is equal to input.

    a = layers.Input(shape=(2,))
    b = base_model(a, training=True)
    model = training_lib.Model(a, b)
    preds = model.predict(x, batch_size=10)
    self.assertEqual(np.min(preds), 0.)  # At least one unit was dropped.

  def test_mask_derived_from_keras_layer(self):
    inputs = input_layer_lib.Input((5, 10))
    mask = input_layer_lib.Input((5,))
    outputs = layers.RNN(layers.LSTMCell(100))(inputs, mask=mask)
    model = training_lib.Model([inputs, mask], outputs)
    model.compile('sgd', 'mse')
    history = model.fit(x=[np.ones((10, 5, 10)),
                           np.zeros((10, 5))],
                        y=np.zeros((10, 100)),
                        batch_size=2)
    # All data is masked, returned values are 0's.
    self.assertEqual(history.history['loss'][0], 0.0)
    history = model.fit(x=[np.ones((10, 5, 10)),
                           np.ones((10, 5))],
                        y=np.zeros((10, 100)),
                        batch_size=2)
    # Data is not masked, returned values are random.
    self.assertGreater(history.history['loss'][0], 0.0)

    model = training_lib.Model.from_config(model.get_config())
    model.compile('sgd', 'mse')
    history = model.fit(x=[np.ones((10, 5, 10)),
                           np.zeros((10, 5))],
                        y=np.zeros((10, 100)),
                        batch_size=2)
    # All data is masked, returned values are 0's.
    self.assertEqual(history.history['loss'][0], 0.0)
    history = model.fit(x=[np.ones((10, 5, 10)),
                           np.ones((10, 5))],
                        y=np.zeros((10, 100)),
                        batch_size=2)
    # Data is not masked, returned values are random.
    self.assertGreater(history.history['loss'][0], 0.0)

  def test_call_arg_derived_from_keras_layer(self):
    class MyAdd(layers.Layer):
      def call(self, x1, x2):  # pylint: disable=arguments-differ
        return x1 + x2

    input1 = input_layer_lib.Input(10)
    input2 = input_layer_lib.Input(10)
    outputs = MyAdd()(input1, input2)
    model = training_lib.Model([input1, input2], outputs)
    model.compile('sgd', 'mse')
    history = model.fit(x=[3 * np.ones((10, 10)), 7 * np.ones((10, 10))],
                        y=10 * np.ones((10, 10)),
                        batch_size=2)
    # Check that second input was correctly added to first.
    self.assertEqual(history.history['loss'][0], 0.0)

    # Check serialization.
    model = training_lib.Model.from_config(model.get_config(),
                                           custom_objects={'MyAdd': MyAdd})
    model.compile('sgd', 'mse')
    history = model.fit(x=[3 * np.ones((10, 10)), 7 * np.ones((10, 10))],
                        y=10 * np.ones((10, 10)),
                        batch_size=2)
    # Check that second input was correctly added to first.
    self.assertEqual(history.history['loss'][0], 0.0)

  def test_only_some_in_first_arg_derived_from_keras_layer_keras_tensors(self):
    class MyAddAll(layers.Layer):
      def call(self, inputs):  # pylint: disable=arguments-differ
        x = inputs[0]
        for inp in inputs[1:]:
          if inp is not None:
            x = x + inp
        return x

    input1 = input_layer_lib.Input(10)
    input2 = input_layer_lib.Input(10)
    layer = MyAddAll()
    outputs = layer([0.0, input1, None, input2, None])
    model = training_lib.Model([input1, input2], outputs)
    self.assertIn(layer, model.layers)
    model.compile('sgd', 'mse')
    history = model.fit(x=[3 * np.ones((10, 10)), 7 * np.ones((10, 10))],
                        y=10 * np.ones((10, 10)),
                        batch_size=2)
    # Check that second input was correctly added to first.
    self.assertEqual(history.history['loss'][0], 0.0)

    # Check serialization.
    model = training_lib.Model.from_config(
        model.get_config(), custom_objects={'MyAddAll': MyAddAll})
    model.compile('sgd', 'mse')
    history = model.fit(x=[3 * np.ones((10, 10)), 7 * np.ones((10, 10))],
                        y=10 * np.ones((10, 10)),
                        batch_size=2)
    # Check that second input was correctly added to first.
    self.assertEqual(history.history['loss'][0], 0.0)

  def test_call_kwarg_dtype_serialization(self):
    class Double(layers.Layer):
      def call(self, x1, dtype=None):  # pylint: disable=arguments-differ
        return math_ops.cast(x1 + x1, dtype=dtype)

    input1 = input_layer_lib.Input(10)
    outputs = Double()(input1, dtype=dtypes.float16)
    model = training_lib.Model([input1], outputs)
    model.compile('sgd', 'mse')
    history = model.fit(x=[3 * np.ones((10, 10))],
                        y=6 * np.ones((10, 10)),
                        batch_size=2)
    # Check that input was correctly doubled.
    self.assertEqual(history.history['loss'][0], 0.0)

    # Check the output dtype
    self.assertEqual(model(array_ops.ones((3, 10))).dtype, dtypes.float16)

    model = training_lib.Model.from_config(model.get_config(),
                                           custom_objects={'Double': Double})
    model.compile('sgd', 'mse')
    history = model.fit(x=[3 * np.ones((10, 10))],
                        y=6 * np.ones((10, 10)),
                        batch_size=2)
    # Check that input was correctly doubled.
    self.assertEqual(history.history['loss'][0], 0.0)

    # Check the output dtype
    self.assertEqual(model(array_ops.ones((3, 10))).dtype, dtypes.float16)

  def test_call_kwarg_nonserializable(self):
    class Double(layers.Layer):
      def call(self, x1, kwarg=None):  # pylint: disable=arguments-differ
        del kwarg
        return x1 + x1

    class NonSerializable(object):
      def __init__(self, foo=None):
        self.foo = foo

    input1 = input_layer_lib.Input(10)
    outputs = Double()(input1, kwarg=NonSerializable())
    model = training_lib.Model([input1], outputs)
    model.compile('sgd', 'mse')
    history = model.fit(x=[3 * np.ones((10, 10))],
                        y=6 * np.ones((10, 10)),
                        batch_size=2)
    # Check that input was correctly doubled.
    self.assertEqual(history.history['loss'][0], 0.0)
    with self.assertRaisesRegex(
        TypeError, 'Layer double was passed non-JSON-serializable arguments.'):
      model.get_config()

  def test_call_nested_arg_derived_from_keras_layer(self):
    class AddAll(layers.Layer):
      def call(self, x1, x2, x3=None):  # pylint: disable=arguments-differ
        out = x1 + x2
        if x3 is not None:
          for t in x3.values():
            out += t
        return out

    input1 = input_layer_lib.Input(10)
    input2 = input_layer_lib.Input(10)
    input3 = input_layer_lib.Input(10)
    outputs = AddAll()(input1,
                       4 * array_ops.ones((1, 10)),
                       x3={
                           'a': input2,
                           'b': input3,
                           'c': 5 * array_ops.ones((1, 10))
                       })
    model = training_lib.Model([input1, input2, input3], outputs)
    model.compile('sgd', 'mse')
    history = model.fit(
        x=[np.ones((10, 10)), 2 * np.ones((10, 10)), 3 * np.ones((10, 10))],
        y=15 * np.ones((10, 10)),
        batch_size=2)
    # Check that all inputs were correctly added.
    self.assertEqual(history.history['loss'][0], 0.0)

    model = training_lib.Model.from_config(model.get_config(),
                                           custom_objects={'AddAll': AddAll})
    model.compile('sgd', 'mse')
    history = model.fit(
        x=[np.ones((10, 10)), 2 * np.ones((10, 10)), 3 * np.ones((10, 10))],
        y=15 * np.ones((10, 10)),
        batch_size=2)
    # Check that all inputs were correctly added.
    self.assertEqual(history.history['loss'][0], 0.0)

  def test_multi_output_model_with_none_masking(self):
    def func(x):
      return [x * 0.2, x * 0.3]

    def output_shape(input_shape):
      return [input_shape, input_shape]

    i = layers.Input(shape=(3, 2, 1))
    o = layers.Lambda(function=func, output_shape=output_shape)(i)

    self.assertEqual(backend.int_shape(o[0]), (None, 3, 2, 1))
    self.assertEqual(backend.int_shape(o[1]), (None, 3, 2, 1))

    o = layers.add(o)
    model = training_lib.Model(i, o)

    i2 = layers.Input(shape=(3, 2, 1))
    o2 = model(i2)
    model2 = training_lib.Model(i2, o2)

    x = np.random.random((4, 3, 2, 1))
    out = model2.predict(x, batch_size=4)
    assert out.shape == (4, 3, 2, 1)
    self.assertAllClose(out, x * 0.2 + x * 0.3, atol=1e-4)

  def test_constant_initializer_with_numpy(self):
    initializer = initializers.Constant(np.ones((3, 2)))
    model = sequential.Sequential()
    model.add(layers.Dense(2, input_shape=(3,),
                           kernel_initializer=initializer))
    model.add(layers.Dense(3))
    model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

    json_str = model.to_json()
    models.model_from_json(json_str)

    if yaml is not None:
      yaml_str = model.to_yaml()
      models.model_from_yaml(yaml_str)

  def test_disconnected_inputs(self):
    input_tensor1 = input_layer_lib.Input(shape=[200], name='a')
    input_tensor2 = input_layer_lib.Input(shape=[10], name='b')
    output_tensor1 = layers.Dense(units=10)(input_tensor1)

    net = functional.Functional(inputs=[input_tensor1, input_tensor2],
                                outputs=[output_tensor1])
    net2 = functional.Functional.from_config(net.get_config())
    self.assertLen(net2.inputs, 2)
    self.assertEqual('a', net2.layers[0].name)
    self.assertEqual('b', net2.layers[1].name)


class DefaultShapeInferenceBehaviorTest(keras_parameterized.TestCase):
  def setUp(self):
    super(DefaultShapeInferenceBehaviorTest, self).setUp()
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 1
    cfg.configure_ipu_system()
    self._ipu_strategy = ipu.ipu_strategy.IPUStrategyV1()
    self._ipu_strategy_scope = self._ipu_strategy.scope()
    self._ipu_strategy_scope.__enter__()

  def tearDown(self):
    self._ipu_strategy_scope.__exit__(None, None, None)
    super(DefaultShapeInferenceBehaviorTest, self).tearDown()

  def _testShapeInference(self, model, input_shape, expected_output_shape):
    input_value = np.random.random(input_shape)
    output_value = model.predict(input_value)
    self.assertEqual(output_value.shape, expected_output_shape)

  def testSingleInputCase(self):
    class LayerWithOneInput(layers.Layer):
      def build(self, input_shape):
        self.w = array_ops.ones(shape=(3, 4))

      def call(self, inputs):  # pylint: disable=arguments-differ
        return backend.dot(inputs, self.w)

    inputs = input_layer_lib.Input(shape=(3,))
    layer = LayerWithOneInput()

    if context.executing_eagerly():
      self.assertEqual(
          layer.compute_output_shape((None, 3)).as_list(), [None, 4])
      # As a side-effect, compute_output_shape builds the layer.
      self.assertTrue(layer.built)
      # We can still query the layer's compute_output_shape with compatible
      # input shapes.
      self.assertEqual(layer.compute_output_shape((6, 3)).as_list(), [6, 4])

    outputs = layer(inputs)
    model = training_lib.Model(inputs, outputs)
    self._testShapeInference(model, (32, 3), (32, 4))

  def testMultiInputOutputCase(self):
    class MultiInputOutputLayer(layers.Layer):
      def build(self, input_shape):
        self.w = array_ops.ones(shape=(3, 4))

      def call(self, inputs):  # pylint: disable=arguments-differ
        a = backend.dot(inputs[0], self.w)
        b = a + inputs[1]
        return [a, b]

    input_a = input_layer_lib.Input(shape=(3,))
    input_b = input_layer_lib.Input(shape=(4,))
    output_a, output_b = MultiInputOutputLayer()([input_a, input_b])
    model = training_lib.Model([input_a, input_b], [output_a, output_b])
    output_a_val, output_b_val = model.predict(
        [np.random.random((32, 3)),
         np.random.random((32, 4))])
    self.assertEqual(output_a_val.shape, (32, 4))
    self.assertEqual(output_b_val.shape, (32, 4))

  def testTrainingArgument(self):
    class LayerWithTrainingArg(layers.Layer):
      def build(self, input_shape):
        self.w = array_ops.ones(shape=(3, 4))

      def call(self, inputs, training):  # pylint: disable=arguments-differ
        del training
        return backend.dot(inputs, self.w)

    inputs = input_layer_lib.Input(shape=(3,))
    outputs = LayerWithTrainingArg()(inputs, training=False)
    model = training_lib.Model(inputs, outputs)
    self._testShapeInference(model, (32, 3), (32, 4))

  def testNoneInShape(self):
    # pylint: disable=abstract-method
    class Model(training_lib.Model):
      def __init__(self):
        super(Model, self).__init__()
        self.conv1 = layers.Conv2D(8, 3)
        self.pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(3)

      def call(self, x):  # pylint: disable=arguments-differ
        x = self.conv1(x)
        x = self.pool(x)
        x = self.fc(x)
        return x

    model = Model()
    model.build(tensor_shape.TensorShape((None, None, None, 1)))
    self.assertTrue(model.built, 'Model should be built')
    self.assertTrue(
        model.weights, 'Model should have its weights created as it '
        'has been built')
    sample_input = array_ops.ones((1, 10, 10, 1))
    output = model(sample_input)
    self.assertEqual(output.shape, (1, 3))

  def testKerasInputAsShape(self):
    # pylint: disable=abstract-method
    class Model(training_lib.Model):
      def __init__(self):
        super(Model, self).__init__()
        self.conv1 = layers.Conv2D(8, 3)
        self.pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(3)

      def call(self, x):  # pylint: disable=arguments-differ
        x = self.conv1(x)
        x = self.pool(x)
        x = self.fc(x)
        return x

    model = Model()
    model.build(input_layer_lib.Input(batch_shape=[None, None, None, 1]))
    self.assertTrue(model.built, 'Model should be built')
    self.assertTrue(
        model.weights, 'Model should have its weights created as it '
        'has been built')
    sample_input = array_ops.ones((1, 10, 10, 1))
    output = model(sample_input)
    self.assertEqual(output.shape, (1, 3))

  def testNoneInShapeWithCompoundModel(self):
    # pylint: disable=abstract-method
    class BasicBlock(training_lib.Model):
      def __init__(self):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(8, 3)
        self.pool = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(3)

      def call(self, x):  # pylint: disable=arguments-differ
        x = self.conv1(x)
        x = self.pool(x)
        x = self.dense(x)
        return x

    # pylint: disable=abstract-method
    class CompoundModel(training_lib.Model):
      def __init__(self):
        super(CompoundModel, self).__init__()
        self.block = BasicBlock()

      def call(self, x):  # pylint: disable=arguments-differ
        x = self.block(x)  # pylint: disable=not-callable
        return x

    model = CompoundModel()
    model.build(tensor_shape.TensorShape((None, None, None, 1)))
    self.assertTrue(model.built, 'Model should be built')
    self.assertTrue(
        model.weights, 'Model should have its weights created as it '
        'has been built')
    sample_input = array_ops.ones((1, 10, 10, 1))
    output = model(sample_input)  # pylint: disable=not-callable
    self.assertEqual(output.shape, (1, 3))

  def testNoneInShapeWithFunctionalAPI(self):
    # pylint: disable=abstract-method
    class BasicBlock(training_lib.Model):
      # Inheriting from layers.Layer since we are calling this layer
      # inside a model created using functional API.

      def __init__(self):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(8, 3)

      def call(self, x):  # pylint: disable=arguments-differ
        x = self.conv1(x)
        return x

    input_layer = layers.Input(shape=(None, None, 1))
    x = BasicBlock()(input_layer)
    x = layers.GlobalAveragePooling2D()(x)
    output_layer = layers.Dense(3)(x)

    model = training_lib.Model(inputs=input_layer, outputs=output_layer)

    model.build(tensor_shape.TensorShape((None, None, None, 1)))
    self.assertTrue(model.built, 'Model should be built')
    self.assertTrue(
        model.weights, 'Model should have its weights created as it '
        'has been built')
    sample_input = array_ops.ones((1, 10, 10, 1))
    output = model(sample_input)
    self.assertEqual(output.shape, (1, 3))

  def test_sequential_as_downstream_of_masking_layer(self):
    inputs = layers.Input(shape=(3, 4))
    x = layers.Masking(mask_value=0., input_shape=(3, 4))(inputs)

    s = sequential.Sequential()
    s.add(layers.Dense(5, input_shape=(4,)))

    x = layers.wrappers.TimeDistributed(s)(x)
    model = training_lib.Model(inputs=inputs, outputs=x)
    model.compile(optimizer='rmsprop', loss='mse')

    model_input = np.random.randint(low=1, high=5,
                                    size=(12, 3, 4)).astype('float32')
    for i in range(4):
      model_input[i, i:, :] = 0.
    model.fit(model_input,
              np.random.random((12, 3, 5)),
              epochs=1,
              batch_size=6)

    if not context.executing_eagerly():
      # Note: this doesn't work in eager due to DeferredTensor/ops compatibility
      # issue.
      mask_outputs = [model.layers[1].compute_mask(model.layers[1].input)]
      mask_outputs += [
          model.layers[2].compute_mask(model.layers[2].input, mask_outputs[-1])
      ]
      func = backend.function([model.input], mask_outputs)
      mask_outputs_val = func([model_input])
      self.assertAllClose(mask_outputs_val[0], np.any(model_input, axis=-1))
      self.assertAllClose(mask_outputs_val[1], np.any(model_input, axis=-1))


class AddLossTest(keras_parameterized.TestCase):
  def setUp(self):
    super().setUp()
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 1
    cfg.configure_ipu_system()
    self._ipu_strategy = ipu.ipu_strategy.IPUStrategyV1()
    self._ipu_strategy_scope = self._ipu_strategy.scope()
    self._ipu_strategy_scope.__enter__()

  def tearDown(self):
    self._ipu_strategy_scope.__exit__(None, None, None)
    super().tearDown()

  def test_add_loss_outside_call_only_loss(self):
    inputs = input_layer_lib.Input((10,))
    mid = layers.Dense(10)(inputs)
    outputs = layers.Dense(1)(mid)
    model = training_lib.Model(inputs, outputs)
    model.add_loss(math_ops.reduce_mean(outputs))
    self.assertLen(model.losses, 1)

    initial_weights = model.get_weights()

    x = np.ones((10, 10))
    model.compile('sgd')
    model.fit(x, batch_size=2, epochs=1)

    model2 = model.from_config(model.get_config())
    model2.compile('sgd')
    model2.set_weights(initial_weights)
    model2.fit(x, batch_size=2, epochs=1)

    # The TFOpLayer and the AddLoss layer are serialized.
    self.assertLen(model2.layers, 5)
    self.assertAllClose(model.get_weights(), model2.get_weights())

  def test_add_loss_outside_call_multiple_losses(self):
    inputs = input_layer_lib.Input((10,))
    x1 = layers.Dense(10)(inputs)
    x2 = layers.Dense(10)(x1)
    outputs = layers.Dense(1)(x2)
    model = training_lib.Model(inputs, outputs)
    model.add_loss(math_ops.reduce_sum(x1 * x2))
    model.add_loss(math_ops.reduce_mean(outputs))
    self.assertLen(model.losses, 2)

    initial_weights = model.get_weights()

    x, y = np.ones((10, 10)), np.ones((10, 1))
    model.compile('sgd', 'mse')
    model.fit(x, y, batch_size=2, epochs=1)

    model2 = model.from_config(model.get_config())
    model2.compile('sgd', 'mse')
    model2.set_weights(initial_weights)
    model2.fit(x, y, batch_size=2, epochs=1)

    self.assertAllClose(model.get_weights(), model2.get_weights())

  def test_add_loss_crossentropy_backtracking(self):
    inputs = input_layer_lib.Input((2,))
    labels = input_layer_lib.Input((1,))
    outputs = layers.Dense(1, activation='sigmoid')(inputs)
    model = functional.Functional([inputs, labels], outputs)
    model.add_loss(losses.binary_crossentropy(labels, outputs))
    model.compile('adam')
    x = np.random.random((2, 2))
    y = np.random.random((2, 1))
    model.fit([x, y], batch_size=2)

    inputs = input_layer_lib.Input((2,))
    labels = input_layer_lib.Input((2,))
    outputs = layers.Dense(2, activation='softmax')(inputs)
    model = functional.Functional([inputs, labels], outputs)
    model.add_loss(losses.categorical_crossentropy(labels, outputs))
    model.compile('adam')
    x = np.random.random((2, 2))
    y = np.random.random((2, 2))
    model.fit([x, y], batch_size=2)

    inputs = input_layer_lib.Input((2,))
    labels = input_layer_lib.Input((1,), dtype='int32')
    outputs = layers.Dense(2, activation='softmax')(inputs)
    model = functional.Functional([inputs, labels], outputs)
    model.add_loss(losses.sparse_categorical_crossentropy(labels, outputs))
    model.compile('adam')
    x = np.random.random((2, 2))
    y = np.random.randint(0, 2, size=(2, 1)).astype(np.int32)
    model.fit([x, y], batch_size=2)


if __name__ == '__main__':
  test.main()
