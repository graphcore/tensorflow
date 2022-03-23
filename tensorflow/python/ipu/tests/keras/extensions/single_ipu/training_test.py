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
"""Tests for training routines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import sys

from absl.testing import parameterized
import numpy as np
import six

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers as layers_module
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizer_v2
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training as training_module
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.training.rmsprop import RMSPropOptimizer
from tensorflow.python import ipu


class TrainingTest(keras_parameterized.TestCase):
  def setUp(self):
    super(TrainingTest, self).setUp()
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
    super(TrainingTest, self).tearDown()

  @keras_parameterized.run_with_all_model_types()
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_model_instrumentation(self):
    layers = [
        layers_module.Dense(10, dtype=np.float64),
        layers_module.Dense(10, dtype=np.float64)
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(1,))

    self.assertTrue(model._instrumented_keras_api)  # pylint: disable=protected-access
    self.assertTrue(model._instrumented_keras_model_class)  # pylint: disable=protected-access
    self.assertFalse(model._instrumented_keras_layer_class)  # pylint: disable=protected-access

  @keras_parameterized.run_with_all_model_types()
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_fit_training_arg(self):
    class ReturnTraining(layers_module.Layer):
      def call(self, inputs, training):  # pylint: disable=arguments-differ
        if training:
          return inputs + array_ops.constant([100], 'float32')
        return inputs + array_ops.constant([0], 'float32')

    model = sequential.Sequential([ReturnTraining()])
    model.compile('sgd', 'mse')
    hist = model.fit(x=np.array([0.]), y=np.array([0.]), batch_size=1)
    self.assertAllClose(hist.history['loss'][0], 10000)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  @parameterized.named_parameters(
      ('train_on_batch', 'train_on_batch'),
      ('test_on_batch', 'test_on_batch'),
      ('predict_on_batch', 'predict_on_batch'),
      ('fit', 'fit'),
      ('evaluate', 'evaluate'),
      ('predict', 'predict'),
  )
  def test_disallow_methods_inside_tf_function(self, method_name):
    model = sequential.Sequential([layers_module.Dense(1)])
    model.compile('sgd', 'mse')

    @def_function.function
    def my_fn():
      getattr(model, method_name)(1)

    error_msg = 'inside a `tf.function`'
    with self.assertRaisesRegex(RuntimeError, error_msg):
      my_fn()

  @keras_parameterized.run_all_keras_modes
  def test_fit_and_validate_learning_phase(self):
    class ReturnTraining(layers_module.Layer):
      def call(self, inputs):  # pylint: disable=arguments-differ
        return backend.in_train_phase(lambda: array_ops.ones_like(inputs),
                                      lambda: array_ops.zeros_like(inputs))

    model = sequential.Sequential([ReturnTraining(input_shape=(2,))])
    model.compile('sgd', loss='mae')

    inputs = np.ones((40, 2), dtype=np.float32)
    targets = np.ones((40, 1), dtype=np.float32)

    # Test correctness with `steps_per_epoch`.
    train_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10, drop_remainder=True)
    val_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10, drop_remainder=True)
    history = model.fit(train_dataset,
                        epochs=2,
                        verbose=1,
                        validation_data=val_dataset)

    # The training loss should be 0.0
    self.assertAllClose(history.history['loss'][0], 0.0)
    # The validation loss should be 1.0.
    self.assertAllClose(history.history['val_loss'][0], 1.0)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_fit_and_validate_training_arg(self):
    class ReturnTraining(layers_module.Layer):
      def call(self, inputs, training=None):  # pylint: disable=arguments-differ
        return backend.in_train_phase(lambda: array_ops.ones_like(inputs),
                                      lambda: array_ops.zeros_like(inputs),
                                      training=training)

    model = sequential.Sequential([ReturnTraining(input_shape=(2,))])
    model.compile('sgd', loss='mae')

    inputs = np.ones((40, 2), dtype=np.float32)
    targets = np.ones((40, 1), dtype=np.float32)

    # Test correctness with `steps_per_epoch`.
    train_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10, drop_remainder=True)
    val_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10, drop_remainder=True)
    history = model.fit(train_dataset,
                        epochs=2,
                        verbose=1,
                        validation_data=val_dataset)

    # The training loss should be 0.0
    self.assertAllClose(history.history['loss'][0], 0.0)
    # The validation loss should be 1.0.
    self.assertAllClose(history.history['val_loss'][0], 1.0)

  @keras_parameterized.run_with_all_model_types()
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_target_dtype_matches_output(self):
    def loss_fn(labels, preds):
      self.assertEqual(labels.dtype, preds.dtype)
      return labels - preds

    layers = [
        layers_module.Dense(10, dtype=np.float16),
        layers_module.Dense(10, dtype=np.float16)
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(1,))
    inputs = np.ones(32, dtype=np.float16)
    targets = np.ones(32, dtype=np.float16)
    model.compile('sgd', loss=loss_fn)
    model.train_on_batch(inputs, targets)
    model.test_on_batch(inputs, targets)
    self.assertEqual(model.predict(inputs).dtype, np.float16)

  @keras_parameterized.run_with_all_model_types()
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_fit_and_validate_nested_training_arg(self):
    class NestedReturnTraining(layers_module.Layer):
      def call(self, inputs, training=None):  # pylint: disable=arguments-differ
        return backend.in_train_phase(lambda: array_ops.ones_like(inputs),
                                      lambda: array_ops.zeros_like(inputs),
                                      training=training)

    class ReturnTraining(layers_module.Layer):
      def __init__(self, input_shape=None, **kwargs):
        super(ReturnTraining, self).__init__(input_shape=input_shape, **kwargs)
        self._nested_layer = None

      def build(self, input_shape):
        del input_shape
        self._nested_layer = NestedReturnTraining()
        self.built = True

      def call(self, inputs):  # pylint: disable=arguments-differ
        return self._nested_layer(inputs)

    model = sequential.Sequential([ReturnTraining(input_shape=(2,))])
    model.compile('sgd', loss='mae')

    inputs = np.ones((40, 2), dtype=np.float32)
    targets = np.ones((40, 1), dtype=np.float32)

    # Test correctness with `steps_per_epoch`.
    train_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10, drop_remainder=True)
    val_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10, drop_remainder=True)
    history = model.fit(train_dataset,
                        epochs=2,
                        verbose=1,
                        validation_data=val_dataset)

    # The training loss should be 0.0
    self.assertAllClose(history.history['loss'][0], 0.0)
    # The validation loss should be 1.0.
    self.assertAllClose(history.history['val_loss'][0], 1.0)

  @keras_parameterized.run_with_all_model_types(
      exclude_models=['sequential', 'subclass'])
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_fit_on_arrays(self):
    input_a = layers_module.Input(shape=(3,), name='input_a')
    input_b = layers_module.Input(shape=(3,), name='input_b')

    dense = layers_module.Dense(4, name='dense')
    dropout = layers_module.Dropout(0.5, name='dropout')
    branch_a = [input_a, dense]
    branch_b = [input_b, dense, dropout]

    model = testing_utils.get_multi_io_model(branch_a, branch_b)

    optimizer = RMSPropOptimizer(learning_rate=0.001)
    loss = 'mse'
    loss_weights = [1., 0.5]
    model.compile(optimizer,
                  loss,
                  metrics=[metrics_module.CategoricalAccuracy(), 'mae'],
                  loss_weights=loss_weights)

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_d_np = np.random.random((10, 4))
    output_e_np = np.random.random((10, 4))

    # Test fit at different verbosity
    model.fit([input_a_np, input_b_np], [output_d_np, output_e_np],
              epochs=1,
              batch_size=5,
              verbose=0)
    model.fit([input_a_np, input_b_np], [output_d_np, output_e_np],
              epochs=1,
              batch_size=5,
              verbose=1)
    model.fit([input_a_np, input_b_np], [output_d_np, output_e_np],
              epochs=2,
              batch_size=5,
              verbose=2)
    model.train_on_batch([input_a_np, input_b_np], [output_d_np, output_e_np])

    # Test with validation data
    model.fit([input_a_np, input_b_np], [output_d_np, output_e_np],
              validation_data=([input_a_np,
                                input_b_np], [output_d_np, output_e_np]),
              epochs=1,
              batch_size=5,
              verbose=0)
    model.fit([input_a_np, input_b_np], [output_d_np, output_e_np],
              validation_data=([input_a_np,
                                input_b_np], [output_d_np, output_e_np]),
              epochs=2,
              batch_size=5,
              verbose=1)
    model.fit([input_a_np, input_b_np], [output_d_np, output_e_np],
              validation_data=([input_a_np,
                                input_b_np], [output_d_np, output_e_np]),
              epochs=2,
              batch_size=5,
              verbose=2)
    # Test with validation split
    model.fit([input_a_np, input_b_np], [output_d_np, output_e_np],
              epochs=2,
              batch_size=5,
              verbose=0,
              validation_split=0.5)

    if testing_utils.get_model_type() == 'functional':
      # Test with dictionary inputs
      model.fit({
          'input_a': input_a_np,
          'input_b': input_b_np
      }, {
          'dense': output_d_np,
          'dropout': output_e_np
      },
                epochs=1,
                batch_size=5,
                verbose=0)
      model.fit({
          'input_a': input_a_np,
          'input_b': input_b_np
      }, {
          'dense': output_d_np,
          'dropout': output_e_np
      },
                epochs=1,
                batch_size=5,
                verbose=1)
      model.fit({
          'input_a': input_a_np,
          'input_b': input_b_np
      }, {
          'dense': output_d_np,
          'dropout': output_e_np
      },
                validation_data=({
                    'input_a': input_a_np,
                    'input_b': input_b_np
                }, {
                    'dense': output_d_np,
                    'dropout': output_e_np
                }),
                epochs=1,
                batch_size=5,
                verbose=0)
      model.train_on_batch({
          'input_a': input_a_np,
          'input_b': input_b_np
      }, {
          'dense': output_d_np,
          'dropout': output_e_np
      })

    # Test with lists for loss, metrics
    loss = ['mae', 'mse']
    model.compile(optimizer,
                  loss,
                  metrics=[metrics_module.CategoricalAccuracy(), 'mae'])
    model.fit([input_a_np, input_b_np], [output_d_np, output_e_np],
              epochs=1,
              batch_size=5,
              verbose=0)

    # Test with dictionaries for loss, metrics, loss weights
    if testing_utils.get_model_type() == 'functional':
      loss = {'dense': 'mse', 'dropout': 'mae'}
      loss_weights = {'dense': 1., 'dropout': 0.5}
      metrics = {
          'dense': 'mse',
          'dropout': metrics_module.CategoricalAccuracy()
      }
      model.compile(optimizer,
                    loss,
                    metrics=metrics,
                    loss_weights=loss_weights)
    model.fit([input_a_np, input_b_np], [output_d_np, output_e_np],
              epochs=1,
              batch_size=5,
              verbose=0)

    # Build single-input model
    x = layers_module.Input(shape=(3,), name='input_a')
    y = layers_module.Dense(4)(x)
    model = training_module.Model(x, y)
    model.compile(optimizer, loss='mse')
    # This will work
    model.fit([input_a_np], output_d_np, epochs=1, batch_size=10)

    # Test model on a list of floats
    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 4))

    # Test execution on inputs that are lists of scalars.
    # TF2 and TF1 have slightly different semantics:
    if context.executing_eagerly():
      # In TF2 to avoid any ambiguity when there are nested lists
      # the entire input gets converted to a
      # single numpy array (& it only works in the case of a single io model)
      model.fit(np.ndarray.tolist(input_a_np),
                np.ndarray.tolist(input_b_np),
                epochs=2,
                batch_size=5,
                verbose=2)
    else:
      # In TF1 there was logic to try disambiguating between the individual
      # inputs when lists are nested. This allowed multi-io functional models
      # to support lists of scalars as input, but it caused ambiguity issues
      # for subclass models & made it trickier to pass multi-dimensional inputs
      # as lists of scalars to single io models. This was an excessive amount
      # of complexity for what boiled down to a convenience method we were
      # mainly just using for writing tests.
      model.fit([np.ndarray.tolist(input_a_np)],
                [np.ndarray.tolist(input_b_np)],
                epochs=2,
                batch_size=5,
                verbose=2)

  @keras_parameterized.run_with_all_model_types(
      exclude_models=['sequential', 'subclass'])
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_evaluate_predict_on_arrays(self):
    a = layers_module.Input(shape=(3,), name='input_a')
    b = layers_module.Input(shape=(3,), name='input_b')

    dense = layers_module.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = layers_module.Dropout(0.5, name='dropout')(c)

    model = training_module.Model([a, b], [d, e])

    optimizer = RMSPropOptimizer(learning_rate=0.001)
    loss = 'mse'
    loss_weights = [1., 0.5]
    model.compile(optimizer,
                  loss,
                  metrics=['mae', metrics_module.CategoricalAccuracy()],
                  loss_weights=loss_weights,
                  sample_weight_mode=None)

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_d_np = np.random.random((10, 4))
    output_e_np = np.random.random((10, 4))

    # Test evaluate at different verbosity
    out = model.evaluate([input_a_np, input_b_np], [output_d_np, output_e_np],
                         batch_size=5,
                         verbose=0)
    self.assertEqual(len(out), 7)
    out = model.evaluate([input_a_np, input_b_np], [output_d_np, output_e_np],
                         batch_size=5,
                         verbose=1)
    self.assertEqual(len(out), 7)
    out = model.evaluate([input_a_np, input_b_np], [output_d_np, output_e_np],
                         batch_size=5,
                         verbose=2)
    self.assertEqual(len(out), 7)
    out = model.test_on_batch([input_a_np, input_b_np],
                              [output_d_np, output_e_np])
    self.assertEqual(len(out), 7)

    # Test evaluate with dictionary inputs
    model.evaluate({
        'input_a': input_a_np,
        'input_b': input_b_np
    }, {
        'dense': output_d_np,
        'dropout': output_e_np
    },
                   batch_size=5,
                   verbose=0)
    model.evaluate({
        'input_a': input_a_np,
        'input_b': input_b_np
    }, {
        'dense': output_d_np,
        'dropout': output_e_np
    },
                   batch_size=5,
                   verbose=1)

    # Test predict
    out = model.predict([input_a_np, input_b_np], batch_size=5)
    self.assertEqual(len(out), 2)
    out = model.predict({
        'input_a': input_a_np,
        'input_b': input_b_np
    },
                        batch_size=10)
    self.assertEqual(len(out), 2)
    out = model.predict_on_batch({
        'input_a': input_a_np,
        'input_b': input_b_np
    })
    self.assertEqual(len(out), 2)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_that_trainable_disables_updates(self):
    val_a = np.random.random((32, 4))
    val_out = np.random.random((32, 4))

    a = layers_module.Input(shape=(4,))
    layer = layers_module.BatchNormalization(input_shape=(4,))
    b = layer(a)
    model = training_module.Model(a, b)

    model.trainable = False
    if not ops.executing_eagerly_outside_functions():
      self.assertEmpty(model.updates)

    model.compile('sgd', 'mse')
    if not ops.executing_eagerly_outside_functions():
      self.assertEmpty(model.updates)

    x1 = model.predict(val_a)
    model.train_on_batch(val_a, val_out)
    x2 = model.predict(val_a)
    self.assertAllClose(x1, x2, atol=1e-7)

    model.trainable = True
    model.compile('sgd', 'mse')
    if not ops.executing_eagerly_outside_functions():
      self.assertAllGreater(len(model.updates), 0)

    model.train_on_batch(val_a, val_out)
    x2 = model.predict(val_a)
    assert np.abs(np.sum(x1 - x2)) > 1e-5

    layer.trainable = False
    model.compile('sgd', 'mse')
    if not ops.executing_eagerly_outside_functions():
      self.assertEmpty(model.updates)

    x1 = model.predict(val_a)
    model.train_on_batch(val_a, val_out)
    x2 = model.predict(val_a)
    self.assertAllClose(x1, x2, atol=1e-7)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_weight_deduplication(self):
    class WatchingLayer(layers_module.Layer):
      def __init__(self, dense_to_track):
        # This will cause the kernel and bias to be double counted, effectively
        # doubling the learning rate if weights are not deduped.
        self._kernel = dense_to_track.kernel
        self._bias = dense_to_track.bias
        super(WatchingLayer, self).__init__()

    inp = layers_module.Input(shape=(1,))
    dense_layer = layers_module.Dense(1)
    dense_output = dense_layer(inp)  # This will build the dense kernel

    # Deterministically set weights to make the test repeatable.
    dense_layer.set_weights([np.ones((1, 1)), np.zeros((1,))])
    output = WatchingLayer(dense_layer)(dense_output)

    model = training_module.Model(inp, output)

    # 0.25 is the edge of the radius of convergence for the double apply case.
    # At lr=0.24, the double apply case will very slowly descend while the
    # correct case will drop very quickly.
    model.compile(loss='mse',
                  optimizer=optimizer_v2.gradient_descent.SGD(0.24))

    x = np.ones((64 * 2,))
    y = 4.5 * x - 3.

    history = model.fit(x, y, batch_size=64, epochs=2, verbose=2)

    # If the gradient apply is duplicated then the loss after 2 epochs will
    # be ~0.15, compared to the correct answer of O(1e-7).
    self.assertLess(history.history['loss'][-1], 1e-6)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_logs_passed_to_callbacks(self):
    input_dim = 5
    num_classes = 1

    class TestCallback(Callback):
      def __init__(self):
        super(TestCallback, self).__init__()
        self.epoch_end_logs = None
        self.batch_end_logs = None
        self.epoch_end_call_count = 0
        self.batch_end_call_count = 0

      def on_epoch_end(self, epoch, logs=None):
        self.epoch_end_logs = logs
        self.epoch_end_call_count += 1

      def on_batch_end(self, batch, logs=None):
        self.batch_end_logs = logs
        self.batch_end_call_count += 1

    model = testing_utils.get_small_sequential_mlp(num_hidden=10,
                                                   num_classes=num_classes,
                                                   input_dim=input_dim)
    model.compile(loss='binary_crossentropy',
                  metrics=['acc'],
                  weighted_metrics=['mae'],
                  optimizer=RMSPropOptimizer(learning_rate=0.01))

    np.random.seed(1337)
    (x_train,
     y_train), (_, _) = testing_utils.get_test_data(train_samples=10,
                                                    test_samples=10,
                                                    input_shape=(input_dim,),
                                                    num_classes=num_classes)
    y_train = y_train.astype(np.int32)

    test_callback = TestCallback()
    model.fit(x_train,
              y_train,
              batch_size=2,
              epochs=2,
              verbose=0,
              callbacks=[test_callback],
              validation_data=(x_train, y_train))
    self.assertEqual(test_callback.batch_end_call_count, 10)
    self.assertEqual(test_callback.epoch_end_call_count, 2)

    self.assertSetEqual(set(test_callback.batch_end_logs.keys()),
                        set(['acc', 'loss', 'mae']))
    self.assertSetEqual(
        set(test_callback.epoch_end_logs.keys()),
        set(['acc', 'loss', 'mae', 'val_acc', 'val_loss', 'val_mae']))

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_logging(self):
    mock_stdout = io.BytesIO() if six.PY2 else io.StringIO()
    model = sequential.Sequential()
    model.add(layers_module.Dense(10, activation='relu'))
    model.add(layers_module.Dense(1, activation='sigmoid'))
    model.compile(RMSPropOptimizer(learning_rate=0.001),
                  loss='binary_crossentropy')
    with test.mock.patch.object(sys, 'stdout', mock_stdout):
      model.fit(np.ones((32, 10), 'float32'),
                np.ones((32, 1), 'float32'),
                epochs=10)
    self.assertTrue('Epoch 5/10' in mock_stdout.getvalue())

  def test_training_with_loss_instance(self):
    a = layers_module.Input(shape=(3,), name='input_a')
    b = layers_module.Input(shape=(3,), name='input_b')

    dense = layers_module.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = layers_module.Dropout(0.5, name='dropout')(c)

    model = training_module.Model([a, b], [d, e])
    loss_weights = [1., 0.5]
    model.compile(RMSPropOptimizer(learning_rate=0.001),
                  loss=losses.MeanSquaredError(),
                  metrics=[metrics_module.CategoricalAccuracy(), 'mae'],
                  loss_weights=loss_weights)

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_d_np = np.random.random((10, 4))
    output_e_np = np.random.random((10, 4))

    model.fit([input_a_np, input_b_np], [output_d_np, output_e_np],
              epochs=1,
              batch_size=5)

  def test_calling_subclass_model_on_different_datasets(self):
    # pylint: disable=abstract-method
    class SubclassedModel(training_module.Model):
      def call(self, inputs):  # pylint: disable=arguments-differ
        return inputs * 2

    model = SubclassedModel()
    dataset_one = dataset_ops.Dataset.range(2).batch(
        2, drop_remainder=True).map(lambda x: math_ops.cast(x, np.float32))
    dataset_two = dataset_ops.Dataset.range(3, 10).batch(
        2, drop_remainder=True).map(lambda x: math_ops.cast(x, np.float32))
    self.assertAllEqual([[0], [2]], model.predict(dataset_one, steps=1))
    self.assertAllEqual([[6], [8], [10], [12]],
                        model.predict(dataset_two, steps=2))

  @keras_parameterized.run_with_all_model_types()
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  @parameterized.named_parameters(
      ('default', 1, 4), ('integer_two', 2, 2), ('integer_four', 4, 1),
      ('simple_list', [1, 3, 4], 3), ('duplicated_list', [4, 2, 2], 2))
  def test_validation_freq(self, validation_freq, expected_runs):
    x, y = np.ones((32, 10)), np.ones((32, 1))
    model = testing_utils.get_small_mlp(2, 1, 10)
    model.compile('sgd', 'mse')

    # pylint: disable=super-init-not-called
    class ValCounter(Callback):
      def __init__(self):
        self.val_runs = 0

      def on_test_begin(self, logs=None):
        self.val_runs += 1

    val_counter = ValCounter()
    model.fit(x,
              y,
              epochs=4,
              validation_data=(x, y),
              validation_freq=validation_freq,
              callbacks=[val_counter])
    self.assertEqual(val_counter.val_runs, expected_runs)

  @keras_parameterized.run_with_all_model_types()
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_layer_with_variable_output(self):
    class VariableOutputLayer(layers_module.Layer):
      def build(self, input_shape):
        self.v = self.add_weight('output_var',
                                 shape=(2, 5),
                                 initializer='ones')

      def call(self, inputs):  # pylint: disable=arguments-differ
        del inputs
        return self.v

    model = testing_utils.get_model_from_layers(
        [VariableOutputLayer(), layers_module.Dense(1)], input_shape=(10,))
    model.compile('sgd', 'mse')
    model.fit(np.ones((10, 10)), np.ones((10, 1)), batch_size=2, epochs=5)

    self.assertLen(model.trainable_variables, 3)

  @keras_parameterized.run_with_all_model_types()
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  @testing_utils.enable_v2_dtype_behavior
  def test_model_dtype(self):
    class AssertTypeLayer(layers_module.Layer):
      def call(self, inputs):  # pylint: disable=arguments-differ
        assert inputs.dtype.name == self.dtype, (
            'Input tensor has type %s which does not match assert type %s' %
            (inputs.dtype.name, self.assert_type))
        return inputs + 1.

    for dtype in ('float16', 'float32'):
      model = testing_utils.get_model_from_layers(
          [AssertTypeLayer(dtype=dtype)], input_shape=(10,))
      model.compile('sgd', 'mse')

      x = np.ones((32, 10))
      y = np.ones((32, 10))
      model.fit(x, y)
      model.test_on_batch(x, y)
      model(x)

  @keras_parameterized.run_with_all_model_types()
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  @testing_utils.enable_v2_dtype_behavior
  def test_model_input_dtype(self):
    model = testing_utils.get_small_mlp(1, 10, 10)
    model.compile('sgd', 'mse')
    x = np.ones((10, 10)).astype(np.float64)
    y = np.ones((10, 10)).astype(np.float64)
    dataset = dataset_ops.Dataset.from_tensor_slices(
        (x, y)).batch(2, drop_remainder=True)
    model.fit(dataset)
    self.assertEqual(model._compute_dtype, 'float32')  # pylint: disable=protected-access

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_subclassed_model_with_training_arg(self):
    class LayerWithTrainingArg(layers_module.Layer):
      def call(self, inputs, training=None):  # pylint: disable=arguments-differ
        self.training = training
        return inputs

    # pylint: disable=abstract-method
    class ModelWithTrainingArg(training_module.Model):
      def __init__(self):
        super(ModelWithTrainingArg, self).__init__()
        self.l1 = LayerWithTrainingArg()

      def call(self, inputs, training=None):  # pylint: disable=arguments-differ
        self.training = training
        inputs = self.l1(inputs, training=training)
        return inputs

    x = np.zeros((32, 2))
    model = ModelWithTrainingArg()
    model.compile(loss='mse', optimizer='sgd')
    model.fit(x, x, epochs=1)

    self.assertIs(model.training, True)
    self.assertIs(model.l1.training, True)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_error_when_model_is_not_compiled(self):
    inputs = input_layer.Input(shape=(1,))
    outputs = layers_module.Dense(1)(inputs)
    model = training_module.Model(inputs, outputs)
    with self.assertRaisesRegex(RuntimeError, 'must compile your model'):
      model.fit(np.ones((1, 1)), np.ones((1, 1)))

    # pylint: disable=abstract-method
    class MyModel(training_module.Model):
      def call(self, x):  # pylint: disable=arguments-differ
        self.add_loss(math_ops.reduce_sum(x))
        return x

    model = MyModel()
    with self.assertRaisesRegex(RuntimeError, 'must compile your model'):
      model.fit(np.random.random((32, 1)), epochs=2)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_outputs_are_floats(self):
    x, y = np.ones((32, 1)), np.ones((32, 1))
    model = sequential.Sequential([layers_module.Dense(1)])
    model.compile('sgd', 'mse', metrics=['accuracy'])

    history = model.fit(x, y, epochs=2)
    self.assertIsInstance(history.history['loss'][0], float)
    self.assertIsInstance(history.history['accuracy'][0], float)

    loss, accuracy = model.train_on_batch(x, y)
    self.assertIsInstance(loss, float)
    self.assertIsInstance(accuracy, float)

    loss, accuracy = model.evaluate(x, y)
    self.assertIsInstance(loss, float)
    self.assertIsInstance(accuracy, float)

    loss, accuracy = model.test_on_batch(x, y)
    self.assertIsInstance(loss, float)
    self.assertIsInstance(accuracy, float)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_int_output(self):
    x, y = np.ones((32, 1)), np.ones((32, 1))
    model = sequential.Sequential([layers_module.Dense(1)])

    class MyMetric(metrics_module.Metric):
      def update_state(self, y_true, y_pred, sample_weight=None):  # pylint: disable=arguments-differ
        del y_true, y_pred, sample_weight

      def result(self):
        return array_ops.constant(1, dtype='int32')

    model.compile('sgd', 'mse', metrics=[MyMetric()])
    history = model.fit(x, y, epochs=2)
    self.assertIsInstance(history.history['my_metric'][0], int)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_calling_aggregate_gradient(self):
    class _Optimizer(optimizer_v2.gradient_descent.SGD):
      """Mock optimizer to check if _aggregate_gradient is called."""

      _HAS_AGGREGATE_GRAD = True

      def __init__(self):
        self.aggregate_gradients_called = False
        super(_Optimizer, self).__init__(name='MyOptimizer')

      def _aggregate_gradients(self, grads):  # pylint: disable=arguments-differ
        self.aggregate_gradients_called = True
        return super(_Optimizer, self)._aggregate_gradients(grads)

    mock_optimizer = _Optimizer()

    model = sequential.Sequential()
    model.add(layers_module.Dense(10, activation='relu'))

    model.compile(mock_optimizer, 'mse')
    x, y = np.ones((32, 10)), np.ones((32, 10))
    model.fit(x, y)
    self.assertEqual(model.optimizer.aggregate_gradients_called, True)

    class _OptimizerOverrideApplyGradients(_Optimizer):
      """Override apply_gradients.

      To test the case where the optimizer does not define the
      experimental_aggregate_gradients parameter.
      """

      _HAS_AGGREGATE_GRAD = False

      def apply_gradients(self, grads_and_vars, name=None):  # pylint: disable=useless-super-delegation,arguments-differ
        return super(_OptimizerOverrideApplyGradients,
                     self).apply_gradients(grads_and_vars, name)

    mock_optimizer = _OptimizerOverrideApplyGradients()
    model.compile(mock_optimizer, 'mse')
    x, y = np.ones((32, 10)), np.ones((32, 10))
    model.fit(x, y)
    self.assertEqual(model.optimizer.aggregate_gradients_called, True)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_gradients_are_none(self):
    class DenseWithExtraWeight(layers_module.Dense):
      def build(self, input_shape):
        # Gradients w.r.t. extra_weights are None
        self.extra_weight_1 = self.add_weight('extra_weight_1',
                                              shape=(),
                                              initializer='ones')
        super(DenseWithExtraWeight, self).build(input_shape)
        self.extra_weight_2 = self.add_weight('extra_weight_2',
                                              shape=(),
                                              initializer='ones')

    model = sequential.Sequential([DenseWithExtraWeight(4, input_shape=(4,))])
    # Test clipping can handle None gradients
    opt = optimizer_v2.adam.Adam(clipnorm=1.0, clipvalue=1.0)
    model.compile(opt, 'mse')
    inputs = np.random.normal(size=(64, 4))
    targets = np.random.normal(size=(64, 4))
    old_kernel = model.get_weights()[1]
    model.fit(inputs, targets)
    new_kernel = model.get_weights()[1]
    self.assertNotAllEqual(old_kernel, new_kernel)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_layer_ordering(self):
    class MyLayer(layers_module.Layer):
      pass

    # pylint: disable=abstract-method
    class MyModel(training_module.Model):
      def __init__(self, name):
        super(MyModel, self).__init__(name=name)

        self.weight = variables_lib.Variable(0, name=name)

        self.direct_sublayer = MyLayer(name='direct')
        self.direct_sublayer.d = {'d': MyLayer(name='direct/dict')}

        self.dict_sublayer = {'d': MyLayer(name='dict')}
        self.dict_sublayer['d'].direct = MyLayer(name='dict/direct')

    model = MyModel('model')
    # All sublayers, including self and recursive sublayers.
    self.assertEqual(['model', 'direct', 'direct/dict', 'dict', 'dict/direct'],
                     [l.name for l in model._flatten_layers()])  # pylint: disable=protected-access
    # Only direct sublayers, including those in data structures.
    self.assertEqual(['direct', 'dict'], [l.name for l in model.layers])

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  @parameterized.named_parameters(('numpy_array', 'numpy_array'),
                                  ('dataset_array', 'dataset_array'))
  def test_single_input_no_tuple_wrapping(self, input_type):
    x = np.ones((10, 1))

    if input_type == 'numpy_array':
      batch_size = 5
      expected_data_type = ops.Tensor
    elif input_type == 'dataset_array':
      x = dataset_ops.Dataset.from_tensor_slices(x).batch(5,
                                                          drop_remainder=True)
      batch_size = None
      expected_data_type = ops.Tensor

    test_case = self

    # pylint: disable=abstract-method
    class MyModel(training_module.Model):
      def train_step(self, data):
        # No tuple wrapping for single x input and no targets.
        test_case.assertIsInstance(data, expected_data_type)
        return super(MyModel, self).train_step(data)

      def test_step(self, data):
        test_case.assertIsInstance(data, expected_data_type)
        return super(MyModel, self).test_step(data)

      def predict_step(self, data):
        test_case.assertIsInstance(data, expected_data_type)
        return super(MyModel, self).predict_step(data)

    inputs = layers_module.Input(shape=(1,), name='my_input')
    outputs = layers_module.Dense(1)(inputs)
    model = MyModel(inputs, outputs)
    model.add_loss(math_ops.reduce_sum(outputs))
    model.compile('sgd', 'mse')
    model.fit(x, batch_size=batch_size)
    model.evaluate(x, batch_size=batch_size)
    model.predict(x, batch_size=batch_size)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  @parameterized.named_parameters(
      ('custom_metrics', False, True), ('compiled_metrics', True, False),
      ('both_compiled_and_custom_metrics', True, True))
  def test_evaluate_with_custom_test_step(self, use_compiled_metrics,
                                          use_custom_metrics):
    # pylint: disable=abstract-method
    class MyModel(training_module.Model):
      def test_step(self, data):
        x, y = data
        pred = self(x)
        metrics = {}
        if use_compiled_metrics:
          self.compiled_metrics.update_state(y, pred)
          self.compiled_loss(y, pred)
          for metric in self.metrics:
            metrics[metric.name] = metric.result()
        if use_custom_metrics:
          custom_metrics = {
              'mean': math_ops.reduce_mean(pred),
              'sum': math_ops.reduce_sum(pred)
          }
          metrics.update(custom_metrics)
        return metrics

    inputs = layers_module.Input((2,))
    outputs = layers_module.Dense(3)(inputs)
    model = MyModel(inputs, outputs)
    if use_compiled_metrics:
      model.compile('adam', 'mse', metrics=['mae', 'mape'])
    else:
      model.compile('adam', 'mse')
    x = np.random.random((32, 2))
    y = np.random.random((32, 3))
    results_list = model.evaluate(x, y)
    results_dict = model.evaluate(x, y, return_dict=True)
    self.assertLen(results_list, len(results_dict))
    if use_compiled_metrics and use_custom_metrics:
      self.assertLen(results_list, 5)
      self.assertEqual(results_list, [
          results_dict['loss'], results_dict['mae'], results_dict['mape'],
          results_dict['mean'], results_dict['sum']
      ])
    if use_compiled_metrics and not use_custom_metrics:
      self.assertLen(results_list, 3)
      self.assertEqual(
          results_list,
          [results_dict['loss'], results_dict['mae'], results_dict['mape']])
    if not use_compiled_metrics and use_custom_metrics:
      self.assertLen(results_list, 2)
      self.assertEqual(results_list,
                       [results_dict['mean'], results_dict['sum']])


class LossWeightingTest(keras_parameterized.TestCase):
  def setUp(self):
    super(LossWeightingTest, self).setUp()
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
    super(LossWeightingTest, self).tearDown()

  @keras_parameterized.run_all_keras_modes
  def test_class_weights(self):
    num_classes = 5
    batch_size = 5
    epochs = 10
    weighted_class = 3
    weight = .5
    train_samples = 1000
    test_samples = 1000
    input_dim = 5
    learning_rate = 0.001

    model = testing_utils.get_small_sequential_mlp(num_hidden=10,
                                                   num_classes=num_classes,
                                                   input_dim=input_dim)
    model.compile(
        loss='categorical_crossentropy',
        metrics=['acc', metrics_module.CategoricalAccuracy()],
        weighted_metrics=['mae', metrics_module.CategoricalAccuracy()],
        optimizer=RMSPropOptimizer(learning_rate=learning_rate))

    np.random.seed(1337)
    (x_train, y_train), (_, y_test) = testing_utils.get_test_data(
        train_samples=train_samples,
        test_samples=test_samples,
        input_shape=(input_dim,),
        num_classes=num_classes)
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    class_weight = dict([(i, 1.) for i in range(num_classes)])
    class_weight[weighted_class] = weight

    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs // 3,
              verbose=0,
              class_weight=class_weight,
              validation_data=(x_train, y_train))
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs // 2,
              verbose=0,
              class_weight=class_weight)
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs // 2,
              verbose=0,
              class_weight=class_weight,
              validation_split=0.1)

    model.train_on_batch(x_train[:batch_size],
                         y_train[:batch_size],
                         class_weight=class_weight)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_temporal_sample_weights(self):
    num_classes = 5
    batch_size = 5
    epochs = 10
    weighted_class = 3
    weight = 10.
    train_samples = 1000
    test_samples = 1000
    input_dim = 5
    timesteps = 3
    learning_rate = 0.001

    with self.cached_session():
      model = sequential.Sequential()
      model.add(
          layers_module.TimeDistributed(layers_module.Dense(num_classes),
                                        input_shape=(timesteps, input_dim)))
      model.add(layers_module.Activation('softmax'))

      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=train_samples,
          test_samples=test_samples,
          input_shape=(input_dim,),
          num_classes=num_classes)
      int_y_train = y_train.copy()
      # convert class vectors to binary class matrices
      y_train = np_utils.to_categorical(y_train, num_classes)
      y_test = np_utils.to_categorical(y_test, num_classes)

      sample_weight = np.ones((y_train.shape[0]))
      sample_weight[int_y_train == weighted_class] = weight

      temporal_x_train = np.reshape(x_train,
                                    (len(x_train), 1, x_train.shape[1]))
      temporal_x_train = np.repeat(temporal_x_train, timesteps, axis=1)
      temporal_x_test = np.reshape(x_test, (len(x_test), 1, x_test.shape[1]))
      temporal_x_test = np.repeat(temporal_x_test, timesteps, axis=1)

      temporal_y_train = np.reshape(y_train,
                                    (len(y_train), 1, y_train.shape[1]))
      temporal_y_train = np.repeat(temporal_y_train, timesteps, axis=1)
      temporal_y_test = np.reshape(y_test, (len(y_test), 1, y_test.shape[1]))
      temporal_y_test = np.repeat(temporal_y_test, timesteps, axis=1)

      temporal_sample_weight = np.reshape(sample_weight,
                                          (len(sample_weight), 1))
      temporal_sample_weight = np.repeat(temporal_sample_weight,
                                         timesteps,
                                         axis=1)

      model.compile(
          RMSPropOptimizer(learning_rate=learning_rate),
          loss='categorical_crossentropy',
          metrics=['acc', metrics_module.CategoricalAccuracy()],
          weighted_metrics=['mae', metrics_module.CategoricalAccuracy()],
          sample_weight_mode='temporal')

      model.fit(temporal_x_train,
                temporal_y_train,
                batch_size=batch_size,
                epochs=epochs // 3,
                verbose=0,
                sample_weight=temporal_sample_weight)
      model.fit(temporal_x_train,
                temporal_y_train,
                batch_size=batch_size,
                epochs=epochs // 3,
                verbose=0,
                sample_weight=temporal_sample_weight,
                validation_split=0.1)

      model.train_on_batch(temporal_x_train[:batch_size],
                           temporal_y_train[:batch_size],
                           sample_weight=temporal_sample_weight[:batch_size])
      model.test_on_batch(temporal_x_train[:batch_size],
                          temporal_y_train[:batch_size],
                          sample_weight=temporal_sample_weight[:batch_size])

  @keras_parameterized.run_with_all_model_types(
      exclude_models=['subclass', 'sequential'])
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_fit_with_incorrect_weights(self):
    input_a = layers_module.Input(shape=(3,), name='input_a')
    input_b = layers_module.Input(shape=(3,), name='input_b')

    dense = layers_module.Dense(2, name='output_1')
    dropout = layers_module.Dropout(0.5, name='output_2')
    branch_a = [input_a, dense]
    branch_b = [input_b, dense, dropout]

    model = testing_utils.get_multi_io_model(branch_a, branch_b)
    model.compile(optimizer='adam', loss='mse')
    x = np.random.random((32, 3))
    y = np.random.random((32, 2))

    with self.assertRaises(ValueError):
      model.fit([x, x], [y, y], epochs=1, sample_weight={'unknown': x})

    with self.assertRaises(ValueError):
      model.fit([x, x], [y, y], epochs=1, class_weight={'unknown': 1})

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_default_sample_weight(self):
    """Verifies that fit works without having to set sample_weight."""
    num_classes = 5
    input_dim = 5
    timesteps = 3
    learning_rate = 0.001

    with self.cached_session():
      model = sequential.Sequential()
      model.add(
          layers_module.TimeDistributed(layers_module.Dense(num_classes),
                                        input_shape=(timesteps, input_dim)))

      x = np.random.random((10, timesteps, input_dim))
      y = np.random.random((10, timesteps, num_classes))
      optimizer = RMSPropOptimizer(learning_rate=learning_rate)

      # sample_weight_mode is a list and mode value is None
      model.compile(optimizer, loss='mse', sample_weight_mode=[None])
      model.fit(x, y, epochs=1, batch_size=10)

      # sample_weight_mode is a list and mode value is `temporal`
      model.compile(optimizer, loss='mse', sample_weight_mode=['temporal'])
      model.fit(x, y, epochs=1, batch_size=10)

      # sample_weight_mode is a dict and mode value is None
      model.compile(optimizer,
                    loss='mse',
                    sample_weight_mode={'time_distributed': None})
      model.fit(x, y, epochs=1, batch_size=10)

      # sample_weight_mode is a dict and mode value is `temporal`
      model.compile(optimizer,
                    loss='mse',
                    sample_weight_mode={'time_distributed': 'temporal'})
      model.fit(x, y, epochs=1, batch_size=10)

      # sample_weight_mode is a not a list/dict and mode value is None
      model.compile(optimizer, loss='mse', sample_weight_mode=None)
      model.fit(x, y, epochs=1, batch_size=10)

      # sample_weight_mode is a not a list/dict and mode value is `temporal`
      model.compile(optimizer, loss='mse', sample_weight_mode='temporal')
      model.fit(x, y, epochs=1, batch_size=10)


@keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                         always_skip_v1=True)
class MaskingTest(keras_parameterized.TestCase):
  def setUp(self):
    super(MaskingTest, self).setUp()
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
    super(MaskingTest, self).tearDown()

  def _get_model(self, input_shape=None):
    layers = [
        layers_module.Masking(mask_value=0),
        layers_module.TimeDistributed(
            layers_module.Dense(1, kernel_initializer='one'))
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape)
    model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
    return model

  @keras_parameterized.run_with_all_model_types()
  def test_masking(self):
    model = self._get_model(input_shape=(2, 1))
    x = np.array([[[1], [1]], [[0], [0]]])
    y = np.array([[[1], [1]], [[1], [1]]])
    loss = model.train_on_batch(x, y)
    self.assertEqual(loss, 0)

  @keras_parameterized.run_with_all_model_types(
      exclude_models=['functional', 'subclass'])
  def test_masking_deferred(self):
    model = self._get_model()
    x = np.array([[[1], [1]], [[0], [0]]])
    y = np.array([[[1], [1]], [[1], [1]]])
    loss = model.train_on_batch(x, y)
    self.assertEqual(loss, 0)

  def test_mask_argument_in_layer(self):
    # Test that the mask argument gets correctly passed to a layer in the
    # functional API.

    class CustomMaskedLayer(layers_module.Layer):
      def __init__(self):
        super(CustomMaskedLayer, self).__init__()
        self.supports_masking = True

      def call(self, inputs, mask=None):  # pylint: disable=arguments-differ
        assert mask is not None
        return inputs

      def compute_output_shape(self, input_shape):
        return input_shape

    x = np.random.random((32, 3))
    inputs = layers_module.Input((3,))
    masked = layers_module.Masking(mask_value=0)(inputs)
    outputs = CustomMaskedLayer()(masked)

    model = training_module.Model(inputs, outputs)
    model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
    y = np.random.random((32, 3))
    model.train_on_batch(x, y)


@keras_parameterized.run_with_all_model_types()
class TestDynamicTrainability(keras_parameterized.TestCase):
  def setUp(self):
    super(TestDynamicTrainability, self).setUp()
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
    super(TestDynamicTrainability, self).tearDown()

  def test_trainable_warning(self):
    x = np.random.random((5, 3))
    y = np.random.random((5, 2))

    model = sequential.Sequential()
    model.add(layers_module.Dense(2, input_dim=3))
    model.trainable = False
    model.compile('rmsprop', 'mse')
    model.trainable = True
    model.train_on_batch(x, y)
    self.assertRaises(Warning)

  def test_trainable_argument(self):
    with self.cached_session():
      x = np.random.random((32, 3))
      y = np.random.random((32, 2))

      model = sequential.Sequential()
      model.add(layers_module.Dense(2, input_dim=3, trainable=False))
      model.compile('rmsprop', 'mse')
      out = model.predict(x)
      model.train_on_batch(x, y)
      out_2 = model.predict(x)
      self.assertAllClose(out, out_2)

      # test with nesting
      inputs = layers_module.Input(shape=(3,))
      output = model(inputs)
      model = training_module.Model(inputs, output)
      model.compile('rmsprop', 'mse')
      out = model.predict(x)
      model.train_on_batch(x, y)
      out_2 = model.predict(x)
      self.assertAllClose(out, out_2)

  def test_layer_trainability_switch(self):
    # with constructor argument, in Sequential
    model = sequential.Sequential()
    model.add(layers_module.Dense(2, trainable=False, input_dim=1))
    self.assertListEqual(model.trainable_weights, [])

    # by setting the `trainable` argument, in Sequential
    model = sequential.Sequential()
    layer = layers_module.Dense(2, input_dim=1)
    model.add(layer)
    self.assertListEqual(model.trainable_weights, layer.trainable_weights)
    layer.trainable = False
    self.assertListEqual(model.trainable_weights, [])

    # with constructor argument, in Model
    x = layers_module.Input(shape=(1,))
    y = layers_module.Dense(2, trainable=False)(x)
    model = training_module.Model(x, y)
    self.assertListEqual(model.trainable_weights, [])

    # by setting the `trainable` argument, in Model
    x = layers_module.Input(shape=(1,))
    layer = layers_module.Dense(2)
    y = layer(x)
    model = training_module.Model(x, y)
    self.assertListEqual(model.trainable_weights, layer.trainable_weights)
    layer.trainable = False
    self.assertListEqual(model.trainable_weights, [])

  def test_model_trainability_switch(self):
    # a non-trainable model has no trainable weights
    x = layers_module.Input(shape=(1,))
    y = layers_module.Dense(2)(x)
    model = training_module.Model(x, y)
    model.trainable = False
    self.assertListEqual(model.trainable_weights, [])

    # same for Sequential
    model = sequential.Sequential()
    model.add(layers_module.Dense(2, input_dim=1))
    model.trainable = False
    self.assertListEqual(model.trainable_weights, [])

  def test_nested_model_trainability(self):
    # a Sequential inside a Model
    inner_model = sequential.Sequential()
    inner_model.add(layers_module.Dense(2, input_dim=1))

    x = layers_module.Input(shape=(1,))
    y = inner_model(x)
    outer_model = training_module.Model(x, y)
    self.assertListEqual(outer_model.trainable_weights,
                         inner_model.trainable_weights)
    inner_model.trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])
    inner_model.trainable = True
    inner_model.layers[-1].trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])

    # a Sequential inside a Sequential
    inner_model = sequential.Sequential()
    inner_model.add(layers_module.Dense(2, input_dim=1))
    outer_model = sequential.Sequential()
    outer_model.add(inner_model)
    self.assertListEqual(outer_model.trainable_weights,
                         inner_model.trainable_weights)
    inner_model.trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])
    inner_model.trainable = True
    inner_model.layers[-1].trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])

    # a Model inside a Model
    x = layers_module.Input(shape=(1,))
    y = layers_module.Dense(2)(x)
    inner_model = training_module.Model(x, y)
    x = layers_module.Input(shape=(1,))
    y = inner_model(x)
    outer_model = training_module.Model(x, y)
    self.assertListEqual(outer_model.trainable_weights,
                         inner_model.trainable_weights)
    inner_model.trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])
    inner_model.trainable = True
    inner_model.layers[-1].trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])

    # a Model inside a Sequential
    x = layers_module.Input(shape=(1,))
    y = layers_module.Dense(2)(x)
    inner_model = training_module.Model(x, y)
    outer_model = sequential.Sequential()
    outer_model.add(inner_model)
    self.assertListEqual(outer_model.trainable_weights,
                         inner_model.trainable_weights)
    inner_model.trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])
    inner_model.trainable = True
    inner_model.layers[-1].trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])

  def test_gan_workflow(self):
    shared_layer = layers_module.BatchNormalization()

    inputs1 = input_layer.Input(10)
    outputs1 = shared_layer(inputs1)
    model1 = training_module.Model(inputs1, outputs1)
    shared_layer.trainable = False
    model1.compile('sgd', 'mse')

    inputs2 = input_layer.Input(10)
    outputs2 = shared_layer(inputs2)
    model2 = training_module.Model(inputs2, outputs2)
    shared_layer.trainable = True
    model2.compile('sgd', 'mse')

    x, y = np.ones((10, 10)), np.ones((10, 10))

    out1_0 = model1.predict_on_batch(x)
    model1.train_on_batch(x, y)
    out1_1 = model1.predict_on_batch(x)
    self.assertAllClose(out1_0, out1_1)

    out2_0 = model2.predict_on_batch(x)
    model2.train_on_batch(x, y)
    out2_1 = model2.predict_on_batch(x)
    self.assertNotAllClose(out2_0, out2_1)

  def test_toggle_value(self):
    input_0 = layers_module.Input(shape=(1,))
    dense_0 = layers_module.Dense(1,
                                  kernel_initializer='ones',
                                  bias_initializer='ones')
    dense_1 = layers_module.Dense(1,
                                  kernel_initializer='ones',
                                  bias_initializer='ones')
    result = layers_module.Add()([dense_0(input_0), dense_1(input_0)])
    model = training_module.Model(input_0, result)
    dense_0.trainable = False
    model.compile('sgd', 'mse')

    x = np.ones((10, 1))
    y = 5 * x + 2
    model.train_on_batch(x, y)
    dense_0.trainable = True
    model.train_on_batch(x, y)
    kernel, bias = dense_0.get_weights()
    self.assertAllEqual([kernel[0, 0], bias[0]], [1., 1.])

    kernel, bias = dense_1.get_weights()
    self.assertAllClose([kernel[0, 0], bias[0]], [1.1176, 1.1176])


class TestTrainingWithDataTensors(keras_parameterized.TestCase):
  def setUp(self):
    super(TestTrainingWithDataTensors, self).setUp()
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
    super(TestTrainingWithDataTensors, self).tearDown()

  def test_training_and_eval_methods_on_symbolic_tensors_multi_io(self):
    a = layers_module.Input(shape=(3,), name='input_a')
    b = layers_module.Input(shape=(3,), name='input_b')

    dense = layers_module.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = layers_module.Dropout(0.5, name='dropout')(c)

    model = training_module.Model([a, b], [d, e])

    optimizer = 'rmsprop'
    loss = 'mse'
    loss_weights = [1., 0.5]
    model.compile(optimizer,
                  loss,
                  metrics=['mae', metrics_module.CategoricalAccuracy()],
                  loss_weights=loss_weights)

    input_a_tf = array_ops.zeros(shape=(32, 3))
    input_b_tf = array_ops.zeros(shape=(32, 3))

    output_d_tf = array_ops.zeros(shape=(32, 4))
    output_e_tf = array_ops.zeros(shape=(32, 4))

    model.fit([input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
              epochs=1,
              steps_per_epoch=2,
              verbose=0)
    model.train_on_batch([input_a_tf, input_b_tf], [output_d_tf, output_e_tf])

    # Test with dictionary inputs
    model.fit({
        'input_a': input_a_tf,
        'input_b': input_b_tf
    }, {
        'dense': output_d_tf,
        'dropout': output_e_tf
    },
              epochs=1,
              steps_per_epoch=2,
              verbose=0)
    model.fit({
        'input_a': input_a_tf,
        'input_b': input_b_tf
    }, {
        'dense': output_d_tf,
        'dropout': output_e_tf
    },
              validation_data=({
                  'input_a': input_a_tf,
                  'input_b': input_b_tf
              }, {
                  'dense': output_d_tf,
                  'dropout': output_e_tf
              }),
              epochs=1,
              steps_per_epoch=2,
              validation_steps=2,
              verbose=0)
    model.train_on_batch({
        'input_a': input_a_tf,
        'input_b': input_b_tf
    }, {
        'dense': output_d_tf,
        'dropout': output_e_tf
    })

    # Test with validation data
    model.fit([input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
              validation_data=([input_a_tf,
                                input_b_tf], [output_d_tf, output_e_tf]),
              epochs=1,
              steps_per_epoch=2,
              validation_steps=2,
              verbose=0)
    # Test evaluation / prediction methods
    model.evaluate([input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
                   steps=2,
                   verbose=0)
    model.predict([input_a_tf, input_b_tf], steps=2)
    model.test_on_batch([input_a_tf, input_b_tf], [output_d_tf, output_e_tf])

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_model_with_partial_loss(self):
    with self.cached_session():
      a = input_layer.Input(shape=(3,), name='input_a')
      a_2 = layers_module.Dense(4, name='dense_1')(a)
      dp = layers_module.Dropout(0.5, name='dropout')
      a_3 = dp(a_2)
      model = training_module.Model(a, [a_2, a_3])

      optimizer = 'rmsprop'
      loss = {'dropout': 'mse'}
      model.compile(optimizer, loss, metrics=['mae'])

      input_a_np = np.random.random((32, 3))
      output_a_np = np.random.random((32, 4))

      # test train_on_batch
      _ = model.train_on_batch(input_a_np, output_a_np)
      _ = model.test_on_batch(input_a_np, output_a_np)
      # fit
      _ = model.fit(input_a_np, output_a_np)
      # evaluate
      _ = model.evaluate(input_a_np, output_a_np)

      # Same without dropout.
      a = input_layer.Input(shape=(3,), name='input_a')
      a_2 = layers_module.Dense(4, name='dense_1')(a)
      a_3 = layers_module.Dense(4, name='dense_2')(a_2)
      model = training_module.Model(a, [a_2, a_3])

      optimizer = 'rmsprop'
      loss = {'dense_2': 'mse'}
      model.compile(optimizer, loss, metrics={'dense_1': 'mae'})

      # test train_on_batch
      _ = model.train_on_batch(input_a_np, output_a_np)
      _ = model.test_on_batch(input_a_np, output_a_np)
      # fit
      _ = model.fit(input_a_np, output_a_np)
      # evaluate
      _ = model.evaluate(input_a_np, output_a_np)


class TestTrainingWithMetrics(keras_parameterized.TestCase):
  """Training tests related to metrics."""
  def setUp(self):
    super(TestTrainingWithMetrics, self).setUp()
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
    super(TestTrainingWithMetrics, self).tearDown()

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_metrics_names(self):
    a = layers_module.Input(shape=(3,), name='input_a')
    b = layers_module.Input(shape=(3,), name='input_b')

    dense = layers_module.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = layers_module.Dropout(0.5, name='dropout')(c)

    model = training_module.Model([a, b], [d, e])

    optimizer = RMSPropOptimizer(learning_rate=0.001)
    metrics = ['mse', metrics_module.BinaryAccuracy()]
    model.compile(optimizer, loss='mae', metrics=metrics)

    mse_metric = 'mse' if context.executing_eagerly() else 'mean_squared_error'
    reference_metric_names = [
        'loss', 'dense_loss', 'dropout_loss', 'dense_' + mse_metric,
        'dense_binary_accuracy', 'dropout_' + mse_metric,
        'dropout_binary_accuracy'
    ]

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_d_np = np.random.random((10, 4))
    output_e_np = np.random.random((10, 4))

    model.fit([input_a_np, input_b_np], [output_d_np, output_e_np],
              epochs=1,
              batch_size=5)
    self.assertEqual(reference_metric_names, model.metrics_names)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_metric_state_reset_between_fit_and_evaluate(self):
    model = sequential.Sequential()
    model.add(layers_module.Dense(3, activation='relu', input_dim=4))
    model.add(layers_module.Dense(1, activation='sigmoid'))
    acc_obj = metrics_module.BinaryAccuracy()
    model.compile(loss='mae',
                  metrics=[acc_obj],
                  optimizer=RMSPropOptimizer(learning_rate=0.001))

    x_train = np.random.random((100, 4))
    y_train = np.random.random((100, 1))
    model.fit(x_train, y_train, batch_size=5, epochs=2)
    self.assertEqual(self.evaluate(acc_obj.count), 100)

    x_test = np.random.random((10, 4))
    y_test = np.random.random((10, 1))
    model.evaluate(x_test, y_test, batch_size=5)
    self.assertEqual(self.evaluate(acc_obj.count), 10)

  @keras_parameterized.run_with_all_model_types(
      exclude_models=['subclass', 'sequential'])
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_metrics_valid_compile_input_formats(self):
    inp_1 = layers_module.Input(shape=(1,), name='input_1')
    inp_2 = layers_module.Input(shape=(1,), name='input_2')
    x = layers_module.Dense(3, kernel_initializer='ones', trainable=False)
    out_1 = layers_module.Dense(1,
                                kernel_initializer='ones',
                                name='output_1',
                                trainable=False)
    out_2 = layers_module.Dense(1,
                                kernel_initializer='ones',
                                name='output_2',
                                trainable=False)

    branch_a = [inp_1, x, out_1]
    branch_b = [inp_2, x, out_2]
    model = testing_utils.get_multi_io_model(branch_a, branch_b)

    # list of metrics.
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=[metrics_module.MeanSquaredError()],
                  weighted_metrics=[metrics_module.MeanSquaredError()])

    # list of list of metrics.
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=[
            metrics_module.MeanSquaredError(),
            [metrics_module.MeanSquaredError(),
             metrics_module.Accuracy()]
        ],
        weighted_metrics=[
            metrics_module.MeanSquaredError(),
            [metrics_module.MeanSquaredError(),
             metrics_module.Accuracy()]
        ])

    # dict of metrics.
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics={
            'output_1':
            metrics_module.MeanSquaredError(),
            'output_2':
            [metrics_module.MeanSquaredError(),
             metrics_module.Accuracy()],
        },
        weighted_metrics={
            'output_1':
            metrics_module.MeanSquaredError(),
            'output_2':
            [metrics_module.MeanSquaredError(),
             metrics_module.Accuracy()],
        })

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_metrics_masking(self):
    np.random.seed(1337)
    model = sequential.Sequential()
    model.add(layers_module.Masking(mask_value=0, input_shape=(2, 1)))
    model.add(
        layers_module.TimeDistributed(
            layers_module.Dense(1, kernel_initializer='ones')))
    model.compile(RMSPropOptimizer(learning_rate=0.001),
                  loss='mse',
                  weighted_metrics=['accuracy'])

    # verify that masking is applied.
    x = np.array([[[1], [1]], [[1], [1]], [[0], [0]]])
    y = np.array([[[1], [1]], [[0], [1]], [[1], [1]]])
    scores = model.train_on_batch(x, y)
    self.assertArrayNear(scores, [0.25, 0.75], 0.1)

    # verify that masking is combined with sample weights.
    w = np.array([3, 2, 4])
    scores = model.train_on_batch(x, y, sample_weight=w)
    self.assertArrayNear(scores, [0.3328, 0.8], 0.001)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_add_metric_with_tensor_on_model(self):
    x = layers_module.Input(shape=(1,))
    y = layers_module.Dense(1, kernel_initializer='ones')(x)
    model = training_module.Model(x, y)
    model.add_metric(math_ops.reduce_sum(y),
                     name='metric_1',
                     aggregation='mean')

    if context.executing_eagerly():
      # This is not a use case in v1 graph mode.
      mean_result = metrics_module.Mean()(y)
      with self.assertRaisesRegex(
          ValueError, 'Expected a symbolic Tensor for the metric value'):
        model.add_metric(mean_result, name='metric_2')
    else:
      with self.assertRaisesRegex(
          ValueError, 'Using the result of calling a `Metric` object '):
        with backend.get_graph().as_default():
          model.add_metric(metrics_module.Mean(name='metric_2')(y))

    model.compile('sgd', loss='mse')

    inputs = np.ones(shape=(10, 1))
    targets = np.ones(shape=(10, 1))
    history = model.fit(inputs,
                        targets,
                        epochs=2,
                        batch_size=5,
                        validation_data=(inputs, targets))
    self.assertEqual(history.history['metric_1'][-1], 5)
    self.assertEqual(history.history['val_metric_1'][-1], 5)

    eval_results = model.evaluate(inputs, targets, batch_size=5)
    self.assertEqual(eval_results[-1], 5)

    model.predict(inputs, batch_size=5)
    model.train_on_batch(inputs, targets)
    model.test_on_batch(inputs, targets)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_add_metric_in_model_call(self):
    # pylint: disable=abstract-method
    class TestModel(training_module.Model):
      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = layers_module.Dense(2, kernel_initializer='ones')
        self.mean = metrics_module.Mean(name='metric_1')

      def call(self, x):  # pylint: disable=arguments-differ
        self.add_metric(math_ops.reduce_sum(x),
                        name='metric_2',
                        aggregation='mean')
        # Provide same name as in the instance created in __init__
        # for eager mode
        self.add_metric(self.mean(x), name='metric_1')
        return self.dense1(x)

    model = TestModel()
    model.compile(loss='mse', optimizer=RMSPropOptimizer(0.01))

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    history = model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))
    self.assertAlmostEqual(history.history['metric_1'][-1], 1, 0)
    self.assertAlmostEqual(history.history['val_metric_1'][-1], 1, 0)
    self.assertAlmostEqual(history.history['metric_2'][-1], 5, 0)
    self.assertAlmostEqual(history.history['val_metric_2'][-1], 5, 0)

    eval_results = model.evaluate(x, y, batch_size=5)
    self.assertAlmostEqual(eval_results[1], 1, 0)
    self.assertAlmostEqual(eval_results[2], 5, 0)

    model.predict(x, batch_size=5)
    model.train_on_batch(x, y)
    model.test_on_batch(x, y)

  @keras_parameterized.run_with_all_model_types()
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_add_metric_in_layer_call(self):
    class TestLayer(layers_module.Layer):
      def build(self, input_shape):
        del input_shape
        self.a = self.add_variable('a', (1, 1),
                                   initializer='ones',
                                   trainable=False)
        self.built = True

      def call(self, inputs):  # pylint: disable=arguments-differ
        self.add_metric(math_ops.reduce_sum(inputs),
                        name='metric_1',
                        aggregation='mean')
        return inputs + 1

    layers = [
        TestLayer(input_shape=(1,)),
        layers_module.Dense(2, kernel_initializer='ones')
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(1,))
    model.compile(loss='mse', optimizer=RMSPropOptimizer(0.01))

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    history = model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))
    self.assertEqual(history.history['metric_1'][-1], 5)
    self.assertAlmostEqual(history.history['val_metric_1'][-1], 5, 0)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_model_metrics_list(self):
    class LayerWithAddMetric(layers_module.Layer):
      def __init__(self):
        super(LayerWithAddMetric, self).__init__()
        self.dense = layers_module.Dense(1, kernel_initializer='ones')

      def __call__(self, inputs):  # pylint: disable=arguments-differ
        outputs = self.dense(inputs)
        self.add_metric(math_ops.reduce_sum(outputs),
                        name='metric_1',
                        aggregation='mean')
        return outputs

    class LayerWithNestedAddMetricLayer(layers_module.Layer):
      def __init__(self):
        super(LayerWithNestedAddMetricLayer, self).__init__()
        self.layer = LayerWithAddMetric()

      def call(self, inputs):  # pylint: disable=arguments-differ
        outputs = self.layer(inputs)
        self.add_metric(math_ops.reduce_sum(outputs),
                        name='metric_2',
                        aggregation='mean')
        return outputs

    x = layers_module.Input(shape=(1,))
    y = LayerWithNestedAddMetricLayer()(x)

    model = training_module.Model(x, y)
    model.add_metric(math_ops.reduce_sum(y),
                     name='metric_3',
                     aggregation='mean')

    if context.executing_eagerly():
      # This is not a use case in v1 graph mode.
      mean_result = metrics_module.Mean()(y)
      with self.assertRaisesRegex(
          ValueError, 'Expected a symbolic Tensor for the metric value'):
        model.add_metric(mean_result, name='metric_4')

    else:
      with self.assertRaisesRegex(
          ValueError, 'Using the result of calling a `Metric` object '):
        with backend.get_graph().as_default():
          model.add_metric(metrics_module.Mean(name='metric_4')(y))

    model.compile('sgd',
                  loss='mse',
                  metrics=[metrics_module.Accuracy('metric_4')])

    model.fit(np.ones((10, 1)), np.ones((10, 1)), batch_size=10)

    # Verify that the metrics added using `compile` and `add_metric` API are
    # included
    self.assertEqual([m.name for m in model.metrics],
                     ['loss', 'metric_4', 'metric_2', 'metric_1', 'metric_3'])

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_model_metrics_list_in_call(self):
    # pylint: disable=abstract-method
    class TestModel(training_module.Model):
      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = layers_module.Dense(2, kernel_initializer='ones')

      def call(self, x):  # pylint: disable=arguments-differ
        self.add_metric(math_ops.reduce_sum(x),
                        name='metric_1',
                        aggregation='mean')
        return self.dense1(x)

    model = TestModel()
    model.compile(loss='mse',
                  optimizer=RMSPropOptimizer(0.01),
                  metrics=[metrics_module.Accuracy('acc')])
    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))

    self.assertEqual([m.name for m in model.metrics],
                     ['loss', 'acc', 'metric_1'])

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_multiple_add_metric_calls(self):
    # pylint: disable=abstract-method
    class TestModel(training_module.Model):
      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = layers_module.Dense(2, kernel_initializer='ones')
        self.mean1 = metrics_module.Mean(name='metric_1')
        self.mean2 = metrics_module.Mean(name='metric_2')

      def call(self, x):  # pylint: disable=arguments-differ
        self.add_metric(self.mean2(x), name='metric_2')
        self.add_metric(self.mean1(x), name='metric_1')
        self.add_metric(math_ops.reduce_sum(x),
                        name='metric_3',
                        aggregation='mean')
        return self.dense1(x)

    model = TestModel()
    self.assertListEqual([m.name for m in model.metrics],
                         ['metric_1', 'metric_2'])
    model.compile(loss='mse', optimizer=RMSPropOptimizer(0.01))

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    history = model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))
    self.assertAlmostEqual(history.history['metric_1'][-1], 1, 0)
    self.assertAlmostEqual(history.history['metric_2'][-1], 1, 0)
    self.assertAlmostEqual(history.history['metric_3'][-1], 5, 0)

    eval_results = model.evaluate(x, y, batch_size=5)
    self.assertArrayNear(eval_results[1:4], [1, 1, 5], 0.1)

    model.predict(x, batch_size=5)
    model.train_on_batch(x, y)
    model.test_on_batch(x, y)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_multiple_add_metric_calls_layer(self):
    class TestLayer(layers_module.Layer):
      def __init__(self):
        super(TestLayer, self).__init__(name='test_layer')
        self.dense1 = layers_module.Dense(2, kernel_initializer='ones')
        self.m1 = metrics_module.Mean(name='m_1')
        self.m2 = [
            metrics_module.Mean(name='m_2'),
            metrics_module.Mean(name='m_3')
        ]
        self.m3 = {
            'mean4': metrics_module.Mean(name='m_4'),
            'mean5': metrics_module.Mean(name='m_5')
        }

      def call(self, x):  # pylint: disable=arguments-differ
        self.add_metric(self.m2[0](x))
        self.add_metric(self.m2[1](x))
        self.add_metric(self.m1(x))
        self.add_metric(self.m3['mean4'](x))
        self.add_metric(self.m3['mean5'](x))
        self.add_metric(math_ops.reduce_sum(x), name='m_6', aggregation='mean')
        return self.dense1(x)

    layer = TestLayer()
    self.assertListEqual([m.name for m in layer.metrics],
                         ['m_1', 'm_2', 'm_3', 'm_4', 'm_5'])

    layer(np.ones((10, 10)))
    self.assertListEqual([m.name for m in layer.metrics],
                         ['m_1', 'm_2', 'm_3', 'm_4', 'm_5', 'm_6'])

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_duplicate_metric_name_in_add_metric(self):
    # pylint: disable=abstract-method
    class TestModel(training_module.Model):
      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = layers_module.Dense(2, kernel_initializer='ones')
        self.mean = metrics_module.Mean(name='metric_1')
        self.mean2 = metrics_module.Mean(name='metric_1')

      def call(self, x):  # pylint: disable=arguments-differ
        self.add_metric(self.mean(x), name='metric_1')
        return self.dense1(x)

    model = TestModel()
    model.compile(loss='mse', optimizer=RMSPropOptimizer(0.01))

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    with self.assertRaisesRegex(
        ValueError,
        'Please provide different names for the metrics you have added. '
        'We found 2 metrics with the name: "metric_1"'):
      model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_add_metric_without_name(self):
    # pylint: disable=abstract-method
    class TestModel(training_module.Model):
      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = layers_module.Dense(2, kernel_initializer='ones')

      def call(self, x):  # pylint: disable=arguments-differ
        self.add_metric(math_ops.reduce_sum(x), aggregation='mean')
        return self.dense1(x)

    model = TestModel()
    model.compile(loss='mse', optimizer=RMSPropOptimizer(0.01))
    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))

    with self.assertRaisesRegex(ValueError,
                                'Please provide a name for your metric like'):
      model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_add_metric_correctness(self):
    inputs = input_layer.Input(shape=(1,))
    targets = input_layer.Input(shape=(1,))

    class Bias(layers_module.Layer):
      def build(self, input_shape):
        self.bias = self.add_variable('bias', (1,), initializer='zeros')
        self.mae = metrics_module.MeanAbsoluteError(name='mae_1')

      def call(self, inputs):  # pylint: disable=arguments-differ
        inputs, targets = inputs
        outputs = inputs + self.bias
        self.add_metric(self.mae(targets, outputs), name='mae_1')
        return outputs

    outputs = Bias()([inputs, targets])
    model = training_module.Model([inputs, targets], outputs)

    model.add_metric(metrics_module.mean_absolute_error(targets, outputs),
                     name='mae_2',
                     aggregation='mean')

    model.compile(loss='mae',
                  optimizer=optimizer_v2.gradient_descent.SGD(0.1),
                  metrics=[metrics_module.MeanAbsoluteError(name='mae_3')])

    x = np.array([[0.], [1.], [2.]])
    y = np.array([[0.5], [2.], [3.5]])
    history = model.fit([x, y], y, batch_size=3, epochs=5)

    expected_val = [1., 0.9, 0.8, 0.7, 0.6]
    for key in ['loss', 'mae_1', 'mae_2', 'mae_3']:
      self.assertAllClose(history.history[key], expected_val, 1e-3)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_add_metric_order(self):
    class MyLayer(layers_module.Layer):
      def call(self, inputs, training=None, mask=None):  # pylint: disable=arguments-differ
        del training
        del mask
        self.add_metric(array_ops.ones([32]) * 2.0,
                        name='two',
                        aggregation='mean')
        return inputs

    # pylint: disable=abstract-method
    class MyModel(training_module.Model):
      def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self._sampler = MyLayer(name='sampler')

      def call(self, inputs, training=None, mask=None):
        z = self._sampler(inputs)
        self.add_metric(array_ops.ones([32]) * 1.0,
                        name='one',
                        aggregation='mean')
        self.add_metric(array_ops.ones([32]) * 3.0,
                        name='three',
                        aggregation='mean')
        return z

    xdata = np.random.uniform(size=[32, 16]).astype(np.float32)
    dataset_train = dataset_ops.Dataset.from_tensor_slices((xdata, xdata))
    dataset_train = dataset_train.batch(32, drop_remainder=True)

    model = MyModel()
    model.compile(optimizer='sgd', loss='mse')
    history = model.fit(dataset_train, epochs=3)
    self.assertDictEqual(
        history.history, {
            'loss': [0.0, 0.0, 0.0],
            'three': [3.0, 3.0, 3.0],
            'two': [2.0, 2.0, 2.0],
            'one': [1.0, 1.0, 1.0]
        })

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_add_metric_aggregation_mean(self):
    # pylint: disable=abstract-method
    class TestModel(training_module.Model):
      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = layers_module.Dense(2, kernel_initializer='ones')

      def call(self, x):  # pylint: disable=arguments-differ
        self.add_metric(math_ops.reduce_sum(x),
                        name='metric_1',
                        aggregation='mean')
        return self.dense1(x)

    model = TestModel()
    model.compile('rmsprop', 'mse')
    model.fit(np.ones(shape=(10, 1)), np.ones(shape=(10, 2)), batch_size=5)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_add_metric_aggregation_none(self):
    # pylint: disable=abstract-method
    class TestModel(training_module.Model):
      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = layers_module.Dense(2, kernel_initializer='ones')
        self.mean = metrics_module.Mean(name='metric_1')

      def call(self, x):  # pylint: disable=arguments-differ
        self.add_metric(self.mean(x), name='metric_1', aggregation=None)
        return self.dense1(x)

    model = TestModel()
    model.compile('rmsprop', 'mse')
    model.fit(np.ones(shape=(10, 1)), np.ones(shape=(10, 2)), batch_size=5)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_model_with_nested_compiled_model(self):
    class LayerWithAddMetric(layers_module.Layer):
      def __init__(self):
        super(LayerWithAddMetric, self).__init__()
        self.dense = layers_module.Dense(1, kernel_initializer='ones')

      def call(self, inputs):  # pylint: disable=arguments-differ
        outputs = self.dense(inputs)
        self.add_metric(math_ops.reduce_sum(outputs),
                        name='mean',
                        aggregation='mean')
        return outputs

    x = layers_module.Input(shape=(1,))
    y = LayerWithAddMetric()(x)

    inner_model = training_module.Model(x, y)
    inner_model.add_metric(math_ops.reduce_sum(y),
                           name='mean1',
                           aggregation='mean')

    inner_model.compile('sgd',
                        loss='mse',
                        metrics=[metrics_module.Accuracy('acc')])
    inner_model.fit(np.ones((10, 1)), np.ones((10, 1)), batch_size=10)

    self.assertEqual([m.name for m in inner_model.metrics],
                     ['loss', 'acc', 'mean', 'mean1'])

    x = layers_module.Input(shape=[1])
    y = inner_model(x)
    outer_model = training_module.Model(x, y)
    outer_model.add_metric(math_ops.reduce_sum(y),
                           name='mean2',
                           aggregation='mean')

    outer_model.compile('sgd',
                        loss='mse',
                        metrics=[metrics_module.Accuracy('acc2')])
    outer_model.fit(np.ones((10, 1)), np.ones((10, 1)), batch_size=10)
    self.assertEqual([m.name for m in outer_model.metrics],
                     ['loss', 'acc2', 'mean', 'mean1', 'mean2'])


class BareUpdateLayer(layers_module.Layer):
  def build(self, input_shape):
    del input_shape
    self.counter = self.add_weight('counter',
                                   dtype='int32',
                                   shape=(),
                                   initializer='zeros',
                                   trainable=False)

  def call(self, inputs):  # pylint: disable=arguments-differ
    state_ops.assign_add(self.counter, 1)
    return math_ops.cast(self.counter, inputs.dtype) * inputs


class LambdaUpdateLayer(layers_module.Layer):
  def build(self, input_shape):
    del input_shape
    self.counter = self.add_weight('counter',
                                   dtype='int32',
                                   shape=(),
                                   initializer='zeros',
                                   trainable=False)

  def call(self, inputs):  # pylint: disable=arguments-differ
    # Make sure update isn't run twice.
    self.add_update(lambda: state_ops.assign_add(self.counter, 1))
    return math_ops.cast(self.counter, inputs.dtype) * inputs


class NestedUpdateLayer(layers_module.Layer):
  def build(self, input_shape):
    self.layer = BareUpdateLayer()
    self.layer.build(input_shape)

  @property
  def counter(self):
    return self.layer.counter

  def call(self, inputs):  # pylint: disable=arguments-differ
    return self.layer(inputs)


class SubgraphUpdateLayer(layers_module.Layer):
  def build(self, input_shape):
    del input_shape
    self.counter = self.add_weight('counter',
                                   dtype='int32',
                                   shape=(),
                                   initializer='zeros',
                                   trainable=False)

  def call(self, inputs, training=None):  # pylint: disable=arguments-differ
    if training is None:
      training = backend.learning_phase()

    if training:
      self.counter.assign(self.counter + 1)
    return inputs


@keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                         always_skip_v1=True)
class TestAutoUpdates(keras_parameterized.TestCase):
  def setUp(self):
    super(TestAutoUpdates, self).setUp()
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
    super(TestAutoUpdates, self).tearDown()

  @keras_parameterized.run_with_all_model_types()
  @parameterized.named_parameters(('bare_update', BareUpdateLayer),
                                  ('lambda_update', LambdaUpdateLayer),
                                  ('nested_update', NestedUpdateLayer))
  def test_updates_in_model(self, layer_builder):
    layer = layer_builder()
    x, y = np.ones((10, 10)), np.ones((10, 1))
    model = testing_utils.get_model_from_layers(
        [layer, layers_module.Dense(1)], input_shape=(10,))
    model.compile('sgd', 'mse')
    model.fit(x, y, batch_size=2, epochs=1)
    self.assertEqual(self.evaluate(layer.counter), 5)

  @keras_parameterized.run_with_all_model_types()
  def test_lambda_updates_trainable_false(self):
    x, y = np.ones((10, 10)), np.ones((10, 1))
    layer = LambdaUpdateLayer()
    model = testing_utils.get_model_from_layers(
        [layer, layers_module.Dense(1)], input_shape=(10,))
    model.compile('sgd', 'mse')
    model.fit(x, y, batch_size=2, epochs=1)
    self.assertEqual(self.evaluate(layer.counter), 5)
    layer.trainable = False
    model.compile('sgd', 'mse')
    model.fit(x, y, batch_size=2, epochs=1)
    self.assertEqual(self.evaluate(layer.counter), 5)

  @keras_parameterized.run_with_all_model_types()
  def test_subgraph_updates_in_model(self):
    layer = SubgraphUpdateLayer()
    x, y = np.ones((10, 10)), np.ones((10, 1))
    model = testing_utils.get_model_from_layers(
        [layer, layers_module.Dense(1)], input_shape=(10,))
    model.compile('sgd', 'mse')
    model.fit(x, y, batch_size=2, epochs=1)
    self.assertEqual(self.evaluate(layer.counter), 5)

  @parameterized.named_parameters(('bare_update', BareUpdateLayer),
                                  ('lambda_update', LambdaUpdateLayer),
                                  ('nested_update', NestedUpdateLayer))
  def test_updates_standalone_layer(self, layer_builder):
    layer = layer_builder()
    y = layer(np.ones((10, 10)))
    self.evaluate(layer.counter.initializer)
    self.evaluate(y)
    self.assertEqual(self.evaluate(layer.counter), 1)

  def test_trainable_false_standalone_layer(self):
    layer = LambdaUpdateLayer()
    y = layer(np.ones((10, 10)))
    self.evaluate(layer.counter.initializer)
    self.evaluate(y)
    self.assertEqual(self.evaluate(layer.counter), 1)
    layer.trainable = False
    y = layer(np.ones((10, 10)))
    self.evaluate(y)
    self.assertEqual(self.evaluate(layer.counter), 1)

  @keras_parameterized.run_with_all_model_types()
  def test_batchnorm_trainable_false(self):
    bn = layers_module.BatchNormalization()
    model = testing_utils.get_model_from_layers(
        [bn, layers_module.Dense(1)], input_shape=(10,))
    bn.trainable = False
    model.compile('sgd', 'mse')
    x, y = np.ones((10, 10)), np.ones((10, 1))
    model.fit(x, y, batch_size=2, epochs=1)
    self.assertAllEqual(self.evaluate(bn.moving_mean), np.zeros((10,)))
    self.assertAllEqual(self.evaluate(bn.moving_variance), np.ones((10,)))


if __name__ == '__main__':
  test.main()
