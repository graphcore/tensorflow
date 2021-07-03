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
"""Tests for Keras callbacks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import json
import os
import re
import shutil
import sys
import threading
import time
import unittest

import numpy as np

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import save_options as save_options_lib
from tensorflow.python.training import adam
from tensorflow.python.training.saving import checkpoint_options as checkpoint_options_lib
from tensorflow.python import ipu

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None

TRAIN_SAMPLES = 10
TEST_SAMPLES = 10
NUM_CLASSES = 2
INPUT_DIM = 3
NUM_HIDDEN = 5
BATCH_SIZE = 5


class Counter(keras.callbacks.Callback):
  """Counts the number of times each callback method was run.

  Attributes:
    method_counts: dict. Contains the counts of time  each callback method was
      run.
  """
  def __init__(self):  # pylint: disable=super-init-not-called
    self.method_counts = collections.defaultdict(int)
    methods_to_count = [
        'on_batch_begin', 'on_batch_end', 'on_epoch_begin', 'on_epoch_end',
        'on_predict_batch_begin', 'on_predict_batch_end', 'on_predict_begin',
        'on_predict_end', 'on_test_batch_begin', 'on_test_batch_end',
        'on_test_begin', 'on_test_end', 'on_train_batch_begin',
        'on_train_batch_end', 'on_train_begin', 'on_train_end'
    ]
    for method_name in methods_to_count:
      setattr(self, method_name,
              self.wrap_with_counts(method_name, getattr(self, method_name)))

  def wrap_with_counts(self, method_name, method):
    def _call_and_count(*args, **kwargs):
      self.method_counts[method_name] += 1
      return method(*args, **kwargs)

    return _call_and_count


def _get_numpy():
  return np.ones((10, 10)), np.ones((10, 1))


@keras_parameterized.run_with_all_model_types(exclude_models='subclass')
@keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                         always_skip_v1=True)
class CallbackCountsTest(keras_parameterized.TestCase):
  def setUp(self):
    super(CallbackCountsTest, self).setUp()
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
    super(CallbackCountsTest, self).tearDown()

  def _check_counts(self, counter, expected_counts):
    """Checks that the counts registered by `counter` are those expected."""
    for method_name, expected_count in expected_counts.items():
      self.assertEqual(counter.method_counts[method_name],
                       expected_count,
                       msg='For method {}: expected {}, got: {}'.format(
                           method_name, expected_count,
                           counter.method_counts[method_name]))

  def _get_model(self):
    layers = [
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(10,))
    model.compile(adam.AdamOptimizer(0.001),
                  'binary_crossentropy',
                  run_eagerly=testing_utils.should_run_eagerly())
    return model

  def test_callback_hooks_are_called_in_evaluate(self):
    x, y = _get_numpy()

    model = self._get_model()
    counter = Counter()
    model.evaluate(x, y, batch_size=2, steps=5, callbacks=[counter])
    self._check_counts(
        counter, {
            'on_test_batch_begin': 5,
            'on_test_batch_end': 5,
            'on_test_begin': 1,
            'on_test_end': 1
        })

  def test_callback_hooks_are_called_in_predict(self):
    x = _get_numpy()[0]

    model = self._get_model()
    counter = Counter()
    model.predict(x, batch_size=2, steps=5, callbacks=[counter])
    self._check_counts(
        counter, {
            'on_predict_batch_begin': 5,
            'on_predict_batch_end': 5,
            'on_predict_begin': 1,
            'on_predict_end': 1
        })

  def test_callback_list_methods(self):
    counter = Counter()
    callback_list = keras.callbacks.CallbackList([counter])

    batch = 0
    callback_list.on_test_batch_begin(batch)
    callback_list.on_test_batch_end(batch)
    callback_list.on_predict_batch_begin(batch)
    callback_list.on_predict_batch_end(batch)

    self._check_counts(
        counter, {
            'on_test_batch_begin': 1,
            'on_test_batch_end': 1,
            'on_predict_batch_begin': 1,
            'on_predict_batch_end': 1
        })


class KerasCallbacksTest(keras_parameterized.TestCase):
  def setUp(self):
    super(KerasCallbacksTest, self).setUp()
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
    super(KerasCallbacksTest, self).tearDown()

  def _get_model(self, input_shape=None):
    layers = [
        keras.layers.Dense(3, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ]
    model = testing_utils.get_model_from_layers(layers,
                                                input_shape=input_shape)
    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=[keras.metrics.CategoricalAccuracy(name='my_acc')],
                  run_eagerly=testing_utils.should_run_eagerly())
    return model

  @keras_parameterized.run_with_all_model_types(exclude_models='subclass')
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_progbar_logging(self):
    model = self._get_model(input_shape=(3,))

    x = array_ops.ones((200, 3))
    y = array_ops.zeros((200, 2))
    dataset = dataset_ops.Dataset.from_tensor_slices(
        (x, y)).batch(10, drop_remainder=True)
    expected_log = r'(.*- loss:.*- my_acc:.*)+'

    with self.captureWritesToStream(sys.stdout) as printed:
      model.fit(dataset, epochs=2, steps_per_epoch=10)
      self.assertRegex(printed.contents(), expected_log)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_callback_warning(self):
    class SleepCallback(keras.callbacks.Callback):
      def on_train_batch_end(self, batch, logs=None):
        time.sleep(0.1)

    model = sequential.Sequential()
    model.add(keras.layers.Dense(1))
    model.compile('sgd',
                  loss='mse',
                  run_eagerly=testing_utils.should_run_eagerly())

    warning_messages = []

    def warning(msg):
      warning_messages.append(msg)

    with test.mock.patch.object(logging, 'warning', warning):
      model.fit(np.ones((16, 1), 'float32'),
                np.ones((16, 1), 'float32'),
                batch_size=1,
                epochs=1,
                callbacks=[SleepCallback()])
    warning_msg = ('Callback method `on_train_batch_end` is slow compared '
                   'to the batch time')
    self.assertIn(warning_msg, '\n'.join(warning_messages))

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_default_callbacks_no_warning(self):
    # Test that without the callback no warning is raised
    model = sequential.Sequential()
    model.add(keras.layers.Dense(1))
    model.compile('sgd',
                  loss='mse',
                  run_eagerly=testing_utils.should_run_eagerly())

    warning_messages = []

    def warning(msg):
      warning_messages.append(msg)

    with test.mock.patch.object(logging, 'warning', warning):
      model.fit(np.ones((16, 1), 'float32'),
                np.ones((16, 1), 'float32'),
                batch_size=1,
                epochs=1)
    self.assertListEqual(warning_messages, [])

  @keras_parameterized.run_with_all_model_types(
      exclude_models=['subclass', 'functional'])
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_progbar_logging_deferred_model_build(self):
    model = self._get_model()
    self.assertFalse(model.built)

    x = array_ops.ones((200, 3))
    y = array_ops.zeros((200, 2))
    dataset = dataset_ops.Dataset.from_tensor_slices(
        (x, y)).batch(10, drop_remainder=True)
    expected_log = r'(.*- loss:.*- my_acc:.*)+'

    with self.captureWritesToStream(sys.stdout) as printed:
      model.fit(dataset, epochs=2, steps_per_epoch=10)
      self.assertRegex(printed.contents(), expected_log)

  @keras_parameterized.run_with_all_model_types(exclude_models='subclass')
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_progbar_logging_validation_data(self):
    model = self._get_model(input_shape=(3,))

    x = array_ops.ones((50, 3))
    y = array_ops.zeros((50, 2))
    training_dataset = dataset_ops.Dataset.from_tensor_slices(
        (x, y)).batch(10, drop_remainder=True)
    val_dataset = dataset_ops.Dataset.from_tensor_slices(
        (x, y)).batch(10, drop_remainder=True)
    expected_log = r'(.*5/5.*- loss:.*- my_acc:.*- val_loss:.*- val_my_acc:.*)+'

    with self.captureWritesToStream(sys.stdout) as printed:
      model.fit(training_dataset, epochs=2, validation_data=val_dataset)
      self.assertRegex(printed.contents(), expected_log)

  @keras_parameterized.run_with_all_model_types(exclude_models='subclass')
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_progbar_logging_validation_split(self):
    model = self._get_model(input_shape=(3,))

    x = np.ones((100, 3))
    y = np.zeros((100, 2))
    expected_log = (
        r'(?s).*1/2.*8/8.*- loss:.*- my_acc:.*- val_loss:.*- val_my_acc:'
        r'.*2/2.*8/8.*- loss:.*- my_acc:.*- val_loss:.*- val_my_acc:.*')

    with self.captureWritesToStream(sys.stdout) as printed:
      model.fit(x, y, batch_size=10, epochs=2, validation_split=0.2)
      self.assertRegex(printed.contents(), expected_log)

  @keras_parameterized.run_with_all_model_types(exclude_models='subclass')
  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_progbar_logging_training_validation(self):
    model = self._get_model(input_shape=(2,))

    def generator():
      for _ in range(100):
        yield [1, 1], 1

    training = dataset_ops.Dataset \
        .from_generator(
            generator=generator,
            output_types=('float64', 'float64'),
            output_shapes=([2], [])) \
        .batch(2, drop_remainder=True) \
        .repeat()
    validation = dataset_ops.Dataset \
        .from_generator(
            generator=generator,
            output_types=('float64', 'float64'),
            output_shapes=([2], [])) \
        .batch(2, drop_remainder=True)
    expected_log = (
        r'(?s).*1/2.*20/20.*- loss:.*- my_acc:.*- val_loss:.*- val_my_acc:'
        r'.*2/2.*20/20.*- loss:.*- my_acc:.*- val_loss:.*- val_my_acc:.*')

    with self.captureWritesToStream(sys.stdout) as printed:
      model.fit(x=training,
                validation_data=validation,
                epochs=2,
                steps_per_epoch=20,
                validation_steps=10)
      self.assertRegex(printed.contents(), expected_log)

  @keras_parameterized.run_with_all_model_types(exclude_models='subclass')
  def test_ModelCheckpoint(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    layers = [
        keras.layers.Dense(NUM_HIDDEN, input_dim=INPUT_DIM, activation='relu'),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(10,))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

    filepath = os.path.join(temp_dir, 'checkpoint.h5')
    (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=NUM_CLASSES)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    # case 1
    monitor = 'val_loss'
    save_best_only = False
    mode = 'auto'

    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(NUM_HIDDEN, input_dim=INPUT_DIM, activation='relu'))
    model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    cbks = [
        keras.callbacks.ModelCheckpoint(filepath,
                                        monitor=monitor,
                                        save_best_only=save_best_only,
                                        mode=mode)
    ]
    model.fit(x_train,
              y_train,
              batch_size=BATCH_SIZE,
              validation_data=(x_test, y_test),
              callbacks=cbks,
              epochs=1,
              verbose=0)
    assert os.path.exists(filepath)
    os.remove(filepath)

    # case 2
    mode = 'min'
    cbks = [
        keras.callbacks.ModelCheckpoint(filepath,
                                        monitor=monitor,
                                        save_best_only=save_best_only,
                                        mode=mode)
    ]
    model.fit(x_train,
              y_train,
              batch_size=BATCH_SIZE,
              validation_data=(x_test, y_test),
              callbacks=cbks,
              epochs=1,
              verbose=0)
    assert os.path.exists(filepath)
    os.remove(filepath)

    # case 3
    mode = 'max'
    monitor = 'val_acc'
    cbks = [
        keras.callbacks.ModelCheckpoint(filepath,
                                        monitor=monitor,
                                        save_best_only=save_best_only,
                                        mode=mode)
    ]
    model.fit(x_train,
              y_train,
              batch_size=BATCH_SIZE,
              validation_data=(x_test, y_test),
              callbacks=cbks,
              epochs=1,
              verbose=0)
    assert os.path.exists(filepath)
    os.remove(filepath)

    # case 4
    save_best_only = True
    cbks = [
        keras.callbacks.ModelCheckpoint(filepath,
                                        monitor=monitor,
                                        save_best_only=save_best_only,
                                        mode=mode)
    ]
    model.fit(x_train,
              y_train,
              batch_size=BATCH_SIZE,
              validation_data=(x_test, y_test),
              callbacks=cbks,
              epochs=1,
              verbose=0)
    assert os.path.exists(filepath)
    os.remove(filepath)

    # Case: metric not available.
    cbks = [
        keras.callbacks.ModelCheckpoint(filepath,
                                        monitor='unknown',
                                        save_best_only=True)
    ]
    model.fit(x_train,
              y_train,
              batch_size=BATCH_SIZE,
              validation_data=(x_test, y_test),
              callbacks=cbks,
              epochs=1,
              verbose=0)
    # File won't be written.
    assert not os.path.exists(filepath)

    # case 5
    save_best_only = False
    period = 2
    mode = 'auto'

    filepath = os.path.join(temp_dir, 'checkpoint.{epoch:02d}.h5')
    cbks = [
        keras.callbacks.ModelCheckpoint(filepath,
                                        monitor=monitor,
                                        save_best_only=save_best_only,
                                        mode=mode,
                                        period=period)
    ]
    model.fit(x_train,
              y_train,
              batch_size=BATCH_SIZE,
              validation_data=(x_test, y_test),
              callbacks=cbks,
              epochs=4,
              verbose=1)
    assert os.path.exists(filepath.format(epoch=2))
    assert os.path.exists(filepath.format(epoch=4))
    os.remove(filepath.format(epoch=2))
    os.remove(filepath.format(epoch=4))
    assert not os.path.exists(filepath.format(epoch=1))
    assert not os.path.exists(filepath.format(epoch=3))

    # Invalid use: this will raise a warning but not an Exception.
    keras.callbacks.ModelCheckpoint(filepath,
                                    monitor=monitor,
                                    save_best_only=save_best_only,
                                    mode='unknown')

    # Case 6: `ModelCheckpoint` with a combination of `save_freq` and `period`.
    # Though `period` is deprecated, we're testing it for
    # backward-compatibility.
    filepath = os.path.join(temp_dir, 'checkpoint.epoch{epoch:02d}.h5')
    cbks = [
        keras.callbacks.ModelCheckpoint(filepath,
                                        monitor=monitor,
                                        mode=mode,
                                        save_freq='epoch',
                                        period=5)
    ]
    assert not os.path.exists(filepath.format(epoch=0))
    assert not os.path.exists(filepath.format(epoch=5))
    model.fit(x_train,
              y_train,
              batch_size=2,
              validation_data=(x_test, y_test),
              callbacks=cbks,
              epochs=10,
              verbose=1)
    assert not os.path.exists(filepath.format(epoch=1))
    assert not os.path.exists(filepath.format(epoch=2))
    assert not os.path.exists(filepath.format(epoch=3))
    assert not os.path.exists(filepath.format(epoch=4))
    assert os.path.exists(filepath.format(epoch=5))
    assert not os.path.exists(filepath.format(epoch=6))
    assert os.path.exists(filepath.format(epoch=10))
    os.remove(filepath.format(epoch=5))
    os.remove(filepath.format(epoch=10))

    # Case 7: `ModelCheckpoint` with an integer `save_freq`
    filepath = os.path.join(temp_dir, 'checkpoint.epoch{epoch:02d}.h5')
    cbks = [
        keras.callbacks.ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            mode=mode,
            save_freq=15,
            period=100)  # The period should be ignored (this test tests this).
    ]
    assert not os.path.exists(filepath.format(epoch=3))
    model.fit(x_train,
              y_train,
              batch_size=2,
              validation_data=(x_test, y_test),
              callbacks=cbks,
              epochs=10,
              verbose=1)
    assert not os.path.exists(filepath.format(epoch=1))
    assert not os.path.exists(filepath.format(epoch=2))
    assert os.path.exists(filepath.format(epoch=3))
    assert not os.path.exists(filepath.format(epoch=4))
    assert not os.path.exists(filepath.format(epoch=5))
    assert os.path.exists(filepath.format(epoch=6))
    assert not os.path.exists(filepath.format(epoch=7))
    assert not os.path.exists(filepath.format(epoch=8))
    assert os.path.exists(filepath.format(epoch=9))
    os.remove(filepath.format(epoch=3))
    os.remove(filepath.format(epoch=6))
    os.remove(filepath.format(epoch=9))

    # Case 8: `ModelCheckpoint` with valid and invalid save_freq argument.
    with self.assertRaisesRegex(ValueError, 'Unrecognized save_freq'):
      keras.callbacks.ModelCheckpoint(filepath,
                                      monitor=monitor,
                                      save_best_only=save_best_only,
                                      mode=mode,
                                      save_freq='invalid_save_freq')
    # The following should not raise ValueError.
    keras.callbacks.ModelCheckpoint(filepath,
                                    monitor=monitor,
                                    save_best_only=save_best_only,
                                    mode=mode,
                                    save_freq='epoch')
    keras.callbacks.ModelCheckpoint(filepath,
                                    monitor=monitor,
                                    save_best_only=save_best_only,
                                    mode=mode,
                                    save_freq=3)

    # Case 9: `ModelCheckpoint` with valid and invalid `options` argument.
    with self.assertRaisesRegex(TypeError, 'tf.train.CheckpointOptions'):
      keras.callbacks.ModelCheckpoint(filepath,
                                      monitor=monitor,
                                      save_best_only=save_best_only,
                                      save_weights_only=True,
                                      mode=mode,
                                      options=save_options_lib.SaveOptions())
    with self.assertRaisesRegex(TypeError, 'tf.saved_model.SaveOptions'):
      keras.callbacks.ModelCheckpoint(
          filepath,
          monitor=monitor,
          save_best_only=save_best_only,
          save_weights_only=False,
          mode=mode,
          options=checkpoint_options_lib.CheckpointOptions())
    keras.callbacks.ModelCheckpoint(
        filepath,
        monitor=monitor,
        save_best_only=save_best_only,
        save_weights_only=True,
        mode=mode,
        options=checkpoint_options_lib.CheckpointOptions())
    keras.callbacks.ModelCheckpoint(filepath,
                                    monitor=monitor,
                                    save_best_only=save_best_only,
                                    save_weights_only=False,
                                    mode=mode,
                                    options=save_options_lib.SaveOptions())

  def _get_dummy_resource_for_model_checkpoint_testing(self):
    def get_input_datasets():
      # Simple training input.
      train_input = [[1.]] * 16
      train_label = [[0.]] * 16
      ds = dataset_ops.Dataset.from_tensor_slices((train_input, train_label))
      return ds.batch(8, drop_remainder=True)

    # Very simple bias model to eliminate randomness.
    optimizer = gradient_descent.SGD(0.1)
    model = sequential.Sequential()
    model.add(testing_utils.Bias(input_shape=(1,)))
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
    train_ds = get_input_datasets()

    temp_dir = self.get_temp_dir()
    filepath = os.path.join(temp_dir, 'checkpoint.epoch{epoch:02d}.h5')

    # The filepath shouldn't exist at the beginning.
    self.assertFalse(os.path.exists(filepath))
    callback = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                               save_weights_only=True)

    return model, train_ds, callback, filepath

  def _run_load_weights_on_restart_test_common_iterations(self):

    (model, train_ds, callback,
     filepath) = self._get_dummy_resource_for_model_checkpoint_testing()
    initial_epochs = 3
    model.fit(train_ds, epochs=initial_epochs, callbacks=[callback])

    # The files should exist after fitting with callback.
    for epoch in range(initial_epochs):
      self.assertTrue(os.path.exists(filepath.format(epoch=epoch + 1)))
    self.assertFalse(os.path.exists(filepath.format(epoch=initial_epochs + 1)))
    self.assertEqual(
        callback._get_most_recently_modified_file_matching_pattern(filepath),  # pylint: disable=protected-access
        filepath.format(epoch=initial_epochs))

    model.fit(train_ds, epochs=1)
    weights_after_one_more_epoch = model.get_weights()

    # The filepath should continue to exist after fitting without callback.
    for epoch in range(initial_epochs):
      self.assertTrue(os.path.exists(filepath.format(epoch=epoch + 1)))

    return model, train_ds, filepath, weights_after_one_more_epoch

  @staticmethod
  def get_ModelCheckpoint_load_weights_on_restart_true_test(save_weights_only):
    def func(self):
      (model, train_ds, filepath, weights_after_one_more_epoch) = \
        self._run_load_weights_on_restart_test_common_iterations()  # pylint: disable=protected-access

      # Sleep for some short time period ensuring the files are created with
      # a different time (in MacOS OSS the granularity is only 1 second).
      time.sleep(2)
      callback = keras.callbacks.ModelCheckpoint(
          filepath=filepath,
          save_weights_only=save_weights_only,
          load_weights_on_restart=True)
      model.fit(train_ds, epochs=1, callbacks=[callback])
      weights_after_model_restoring_and_one_more_epoch = model.get_weights()

      self.assertEqual(
          callback._get_most_recently_modified_file_matching_pattern(filepath),  # pylint: disable=protected-access
          filepath.format(epoch=1))

      model.fit(train_ds,
                epochs=1,
                callbacks=[
                    keras.callbacks.ModelCheckpoint(
                        filepath=filepath,
                        save_weights_only=save_weights_only,
                        load_weights_on_restart=True)
                ])
      weights_with_one_final_extra_epoch = model.get_weights()

      # Asserting the weights one epoch after initial fitting and another epoch
      # after that are closed, if a ModelCheckpoint with
      # load_weights_on_restart=True is given (so the model is restored at the
      # beginning of training).
      self.assertAllClose(weights_after_one_more_epoch,
                          weights_after_model_restoring_and_one_more_epoch)

      self.assertNotAllClose(weights_after_one_more_epoch,
                             weights_with_one_final_extra_epoch)

    return func

  @staticmethod
  def get_ModelCheckpoint_load_weights_on_restart_false_test(
      save_weights_only):
    def func(self):
      (model, train_ds, filepath, weights_after_one_more_epoch) = \
        self._run_load_weights_on_restart_test_common_iterations()  # pylint: disable=protected-access

      model.fit(train_ds,
                epochs=1,
                callbacks=[
                    keras.callbacks.ModelCheckpoint(
                        filepath=filepath, save_weights_only=save_weights_only)
                ])
      weights_after_model_restoring_and_one_more_epoch = model.get_weights()

      # Asserting the weights one epoch after initial fitting and another epoch
      # after that are different, if a ModelCheckpoint with
      # load_weights_on_restart=False is given (so the model is not restored at
      # the beginning of training).
      self.assertNotAllClose(weights_after_one_more_epoch,
                             weights_after_model_restoring_and_one_more_epoch)

    return func

  test_model_checkpoint_load_weights_on_restart_true_save_weights_only_true = \
        get_ModelCheckpoint_load_weights_on_restart_true_test.__func__(True)

  test_model_checkpoint_load_weights_on_restart_true_save_weights_only_false = \
        get_ModelCheckpoint_load_weights_on_restart_true_test.__func__(False)

  test_model_checkpoint_load_weights_on_restart_false_save_weights_only_true = \
        get_ModelCheckpoint_load_weights_on_restart_false_test.__func__(True)

  test_model_checkpoint_load_weights_on_restart_false_save_weights_only_false \
        = get_ModelCheckpoint_load_weights_on_restart_false_test.__func__(False)

  def test_ModelCheckpoint_override_if_file_exist(self):
    (model, train_ds, filepath, _) = \
      self._run_load_weights_on_restart_test_common_iterations()  # pylint: disable=protected-access

    # Sleep for some short time period to ensure the files are created with
    # a different time (in MacOS OSS the granularity is only 1 second).
    time.sleep(2)
    callback = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                               save_weights_only=True)
    model.load_weights(
        callback._get_most_recently_modified_file_matching_pattern(filepath))  # pylint: disable=protected-access
    weights_before_additional_fit = model.get_weights()
    model.fit(train_ds, epochs=1, callbacks=[callback])
    model.load_weights(
        callback._get_most_recently_modified_file_matching_pattern(filepath))  # pylint: disable=protected-access
    weights_after_additional_fit = model.get_weights()

    self.assertNotAllClose(weights_before_additional_fit,
                           weights_after_additional_fit)

  def test_fit_with_ModelCheckpoint_with_tf_config(self):
    (model, train_ds, callback,
     _) = self._get_dummy_resource_for_model_checkpoint_testing()

    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ['localhost:23333']
        },
        'task': {
            'type': 'worker',
            'index': 0
        }
    })

    # `model.fit()` should work regardless of the presence of `TF_CONFIG`.
    model.fit(train_ds, epochs=1, callbacks=[callback])

  def test_fit_with_ModelCheckpoint_with_dir_as_h5_filepath(self):
    (model, train_ds, callback,
     filepath) = self._get_dummy_resource_for_model_checkpoint_testing()

    temp_dir = self.get_temp_dir()
    filepath = os.path.join(temp_dir, 'temp.h5')

    self.assertFalse(os.path.exists(filepath))
    os.mkdir(filepath)
    self.assertTrue(os.path.exists(filepath))

    callback = keras.callbacks.ModelCheckpoint(filepath=filepath)

    with self.assertRaisesRegex(
        IOError, 'Please specify a non-directory '
        'filepath for ModelCheckpoint.'):
      model.fit(train_ds, epochs=1, callbacks=[callback])

  def test_ModelCheckpoint_with_bad_path_placeholders(self):
    (model, train_ds, callback,
     filepath) = self._get_dummy_resource_for_model_checkpoint_testing()

    temp_dir = self.get_temp_dir()
    filepath = os.path.join(temp_dir, 'chkpt_{epoch:02d}_{mape:.2f}.h5')
    callback = keras.callbacks.ModelCheckpoint(filepath=filepath)

    with self.assertRaisesRegex(KeyError, 'Failed to format this callback '
                                'filepath.*'):
      model.fit(train_ds, epochs=1, callbacks=[callback])

  def test_ModelCheckpoint_nonblocking(self):
    filepath = self.get_temp_dir()
    # Should only cause a sync block when saving is actually performed.
    callback = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                               save_freq=100)
    self.assertTrue(callback._supports_tf_logs)  # pylint: disable=protected-access

    model = keras.Sequential([keras.layers.Dense(1)])
    cb_list = keras.callbacks.CallbackList([callback],
                                           model=model,
                                           epochs=1,
                                           steps=10,
                                           verbose=0)

    tensor = ops.convert_to_tensor_v2_with_dispatch(1.)

    def mock_numpy():
      raise RuntimeError(
          'If this error is seen, ModelCheckpoint is causing a blocking '
          'NumPy conversion even when not checkpointing.')

    tensor.numpy = mock_numpy

    logs = {'metric': tensor}

    cb_list.on_train_begin(logs)
    cb_list.on_epoch_begin(0, logs)
    cb_list.on_train_batch_begin(0, logs)
    cb_list.on_train_batch_end(0, logs)
    cb_list.on_epoch_end(0, logs)
    cb_list.on_train_end(logs)

    cb_list.on_test_begin(logs)
    cb_list.on_test_batch_begin(0, logs)
    cb_list.on_test_batch_end(0, logs)
    cb_list.on_test_end(logs)

    cb_list.on_predict_begin(logs)
    cb_list.on_predict_batch_begin(logs)
    cb_list.on_predict_batch_end(logs)
    cb_list.on_predict_end(logs)

  def test_ProgbarLogger_verbose_2_nonblocking(self):
    # Should only cause a sync block on epoch end methods.
    callback = keras.callbacks.ProgbarLogger(count_mode='steps')
    self.assertTrue(callback._supports_tf_logs)  # pylint: disable=protected-access

    model = keras.Sequential([keras.layers.Dense(1)])
    cb_list = keras.callbacks.CallbackList([callback],
                                           model=model,
                                           epochs=1,
                                           steps=10,
                                           verbose=2)

    tensor = ops.convert_to_tensor_v2_with_dispatch(1.)

    def mock_numpy():
      raise RuntimeError(
          'If this error is seen, ModelCheckpoint is causing a blocking '
          'NumPy conversion even when not checkpointing.')

    tensor.numpy = mock_numpy
    logs = {'metric': tensor}

    cb_list.on_train_begin(logs)
    cb_list.on_epoch_begin(0, logs)
    cb_list.on_train_batch_begin(0, logs)
    cb_list.on_train_batch_end(0, logs)

    cb_list.on_test_begin(logs)
    cb_list.on_test_batch_begin(0, logs)
    cb_list.on_test_batch_end(0, logs)
    cb_list.on_test_end(logs)

    with self.assertRaisesRegex(RuntimeError, 'NumPy conversion'):
      # on_epoch_end should still block.
      cb_list.on_epoch_end(0, logs)
    cb_list.on_train_end(logs)

  def test_EarlyStopping(self):
    with self.cached_session():
      np.random.seed(123)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = np_utils.to_categorical(y_test)
      y_train = np_utils.to_categorical(y_train)
      model = testing_utils.get_small_sequential_mlp(num_hidden=NUM_HIDDEN,
                                                     num_classes=NUM_CLASSES,
                                                     input_dim=INPUT_DIM)
      model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['acc'])

      cases = [('max', 'val_acc'), ('min', 'val_loss'), ('auto', 'val_acc'),
               ('auto', 'loss'), ('unknown', 'unknown')]
      for mode, monitor in cases:
        patience = 0
        cbks = [
            keras.callbacks.EarlyStopping(patience=patience,
                                          monitor=monitor,
                                          mode=mode)
        ]
        model.fit(x_train,
                  y_train,
                  batch_size=BATCH_SIZE,
                  validation_data=(x_test, y_test),
                  callbacks=cbks,
                  epochs=5,
                  verbose=0)

  def test_EarlyStopping_reuse(self):
    with self.cached_session():
      np.random.seed(1337)
      patience = 3
      data = np.random.random((128, 1))
      labels = np.where(data > 0.5, 1, 0).astype(np.int32)
      model = keras.models.Sequential((
          keras.layers.Dense(1, input_dim=1, activation='relu'),
          keras.layers.Dense(1, activation='sigmoid'),
      ))
      model.compile(optimizer='sgd',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
      weights = model.get_weights()

      # This should allow training to go for at least `patience` epochs
      model.set_weights(weights)

      stopper = keras.callbacks.EarlyStopping(monitor='accuracy',
                                              patience=patience)
      hist = model.fit(data, labels, callbacks=[stopper], verbose=0, epochs=20)
      assert len(hist.epoch) >= patience

  def test_EarlyStopping_with_baseline(self):
    with self.cached_session():
      np.random.seed(1337)
      baseline = 0.6
      (data, labels), _ = testing_utils.get_test_data(train_samples=128,
                                                      test_samples=50,
                                                      input_shape=(1,),
                                                      num_classes=NUM_CLASSES)
      labels = labels.astype(np.int32)
      model = testing_utils.get_small_sequential_mlp(num_hidden=1,
                                                     num_classes=1,
                                                     input_dim=1)
      model.compile(optimizer='sgd',
                    loss='binary_crossentropy',
                    metrics=['acc'])

      stopper = keras.callbacks.EarlyStopping(monitor='acc', baseline=baseline)
      hist = model.fit(data, labels, callbacks=[stopper], verbose=0, epochs=20)
      assert len(hist.epoch) == 1

      patience = 3
      stopper = keras.callbacks.EarlyStopping(monitor='acc',
                                              patience=patience,
                                              baseline=baseline)
      hist = model.fit(data, labels, callbacks=[stopper], verbose=0, epochs=20)
      assert len(hist.epoch) >= patience

  def test_LearningRateScheduler(self):
    with self.cached_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = np_utils.to_categorical(y_test)
      y_train = np_utils.to_categorical(y_train)
      model = testing_utils.get_small_sequential_mlp(num_hidden=NUM_HIDDEN,
                                                     num_classes=NUM_CLASSES,
                                                     input_dim=INPUT_DIM)
      model.compile(loss='categorical_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy'])

      cbks = [keras.callbacks.LearningRateScheduler(lambda x: 1. / (1. + x))]
      model.fit(x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=5,
                verbose=0)
      assert (float(keras.backend.get_value(model.optimizer.lr)) -
              0.2) < keras.backend.epsilon()

      cbks = [keras.callbacks.LearningRateScheduler(lambda x, lr: lr / 2)]
      model.compile(loss='categorical_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy'])
      model.fit(x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=2,
                verbose=0)
      assert (float(keras.backend.get_value(model.optimizer.lr)) -
              0.01 / 4) < keras.backend.epsilon()

      cbks = [
          keras.callbacks.LearningRateScheduler(
              lambda epoch, _: learning_rate_schedule.CosineDecay(0.01, 2)
              (epoch))
      ]
      model.compile(loss='categorical_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy'])
      model.fit(x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=2,
                verbose=0)

      cosine_decay_np = 0.5 * (1 + np.cos(np.pi * (1 / 2)))
      decayed_learning_rate = 0.01 * cosine_decay_np

      assert (float(keras.backend.get_value(model.optimizer.lr)) -
              decayed_learning_rate) < keras.backend.epsilon()

  def test_ReduceLROnPlateau(self):
    with self.cached_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = np_utils.to_categorical(y_test)
      y_train = np_utils.to_categorical(y_train)

      def make_model():
        random_seed.set_random_seed(1234)
        np.random.seed(1337)
        model = testing_utils.get_small_sequential_mlp(num_hidden=NUM_HIDDEN,
                                                       num_classes=NUM_CLASSES,
                                                       input_dim=INPUT_DIM)
        model.compile(loss='categorical_crossentropy',
                      optimizer=gradient_descent.SGD(lr=0.1))
        return model

      # TODO(psv): Make sure the callback works correctly when min_delta is
      # set as 0. Test fails when the order of this callback and assertion is
      # interchanged.
      model = make_model()
      cbks = [
          keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.1,
                                            min_delta=0,
                                            patience=1,
                                            cooldown=5)
      ]
      model.fit(x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=2,
                verbose=0)
      self.assertAllClose(float(keras.backend.get_value(model.optimizer.lr)),
                          0.1,
                          atol=1e-4)

      model = make_model()
      # This should reduce the LR after the first epoch (due to high epsilon).
      cbks = [
          keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.1,
                                            min_delta=10,
                                            patience=1,
                                            cooldown=5)
      ]
      model.fit(x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=2,
                verbose=2)
      self.assertAllClose(float(keras.backend.get_value(model.optimizer.lr)),
                          0.01,
                          atol=1e-4)

  def test_ReduceLROnPlateau_backwards_compatibility(self):
    with test.mock.patch.object(logging, 'warning') as mock_log:
      reduce_on_plateau = keras.callbacks.ReduceLROnPlateau(epsilon=1e-13)
      self.assertRegex(str(mock_log.call_args),
                       '`epsilon` argument is deprecated')
    self.assertFalse(hasattr(reduce_on_plateau, 'epsilon'))
    self.assertTrue(hasattr(reduce_on_plateau, 'min_delta'))
    self.assertEqual(reduce_on_plateau.min_delta, 1e-13)

  def test_CSVLogger(self):
    with self.cached_session():
      np.random.seed(1337)
      temp_dir = self.get_temp_dir()
      self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
      filepath = os.path.join(temp_dir, 'log.tsv')

      sep = '\t'
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = np_utils.to_categorical(y_test)
      y_train = np_utils.to_categorical(y_train)

      def make_model():
        np.random.seed(1337)
        model = testing_utils.get_small_sequential_mlp(num_hidden=NUM_HIDDEN,
                                                       num_classes=NUM_CLASSES,
                                                       input_dim=INPUT_DIM)
        model.compile(loss='categorical_crossentropy',
                      optimizer=gradient_descent.SGD(lr=0.1),
                      metrics=['accuracy'])
        return model

      # case 1, create new file with defined separator
      model = make_model()
      cbks = [keras.callbacks.CSVLogger(filepath, separator=sep)]
      model.fit(x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=1,
                verbose=0)

      assert os.path.exists(filepath)
      with open(filepath) as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read())
      assert dialect.delimiter == sep
      del model
      del cbks

      # case 2, append data to existing file, skip header
      model = make_model()
      cbks = [keras.callbacks.CSVLogger(filepath, separator=sep, append=True)]
      model.fit(x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=1,
                verbose=0)

      # case 3, reuse of CSVLogger object
      model.fit(x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=2,
                verbose=0)

      with open(filepath) as csvfile:
        list_lines = csvfile.readlines()
        for line in list_lines:
          assert line.count(sep) == 4
        assert len(list_lines) == 5
        output = ' '.join(list_lines)
        assert len(re.findall('epoch', output)) == 1

      os.remove(filepath)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_TerminateOnNaN(self):
    np.random.seed(1337)
    (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=NUM_CLASSES)

    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    cbks = [keras.callbacks.TerminateOnNaN()]
    model = keras.models.Sequential()
    initializer = keras.initializers.Constant(value=1e5)
    for _ in range(5):
      model.add(
          keras.layers.Dense(2,
                             input_dim=INPUT_DIM,
                             activation='relu',
                             kernel_initializer=initializer))
    model.add(keras.layers.Dense(NUM_CLASSES))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    history = model.fit(x_train,
                        y_train,
                        batch_size=BATCH_SIZE,
                        validation_data=(x_test, y_test),
                        callbacks=cbks,
                        epochs=20)
    loss = history.history['loss']
    self.assertEqual(len(loss), 1)
    self.assertTrue(np.isnan(loss[0]) or np.isinf(loss[0]))

  @unittest.skipIf(
      os.name == 'nt',
      'use_multiprocessing=True does not work on windows properly.')
  def test_LambdaCallback(self):
    with self.cached_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = np_utils.to_categorical(y_test)
      y_train = np_utils.to_categorical(y_train)
      model = keras.models.Sequential()
      model.add(
          keras.layers.Dense(NUM_HIDDEN,
                             input_dim=INPUT_DIM,
                             activation='relu'))
      model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
      model.compile(loss='categorical_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy'])

      # Start an arbitrary process that should run during model
      # training and be terminated after training has completed.
      e = threading.Event()

      def target():
        e.wait()

      t = threading.Thread(target=target)
      t.start()
      cleanup_callback = keras.callbacks.LambdaCallback(
          on_train_end=lambda logs: e.set())

      cbks = [cleanup_callback]
      model.fit(x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=5,
                verbose=0)
      t.join()
      assert not t.is_alive()

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_callback_passed_floats(self):
    class MyCallback(keras.callbacks.Callback):
      def on_batch_end(self, batch, logs=None):
        assert isinstance(batch, int)
        assert isinstance(logs['loss'], float)
        self.on_batch_end_called = True

      def on_epoch_end(self, batch, logs=None):  # pylint: disable=arguments-differ
        assert isinstance(batch, int)
        assert isinstance(logs['loss'], float)
        self.on_epoch_end_called = True

    x, y = np.ones((32, 1)), np.ones((32, 1))
    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile('sgd', 'mse', run_eagerly=testing_utils.should_run_eagerly())

    callback = MyCallback()
    model.fit(x, y, epochs=2, callbacks=[callback])
    self.assertTrue(callback.on_batch_end_called)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_implements_batch_hooks(self):
    class MyCallbackWithBatchHooks(keras.callbacks.Callback):
      def __init__(self):  # pylint: disable=super-init-not-called
        self.train_batches = 0
        self.test_batches = 0
        self.predict_batches = 0

      def on_train_batch_end(self, batch, logs=None):
        self.train_batches += 1

      def on_test_batch_end(self, batch, logs=None):
        self.test_batches += 1

      def on_predict_batch_end(self, batch, logs=None):
        self.predict_batches += 1

    class MyCallbackWithoutBatchHooks(keras.callbacks.Callback):
      def __init__(self):  # pylint: disable=super-init-not-called
        self.epochs = 0

      def on_epoch_end(self, batch, logs=None):  # pylint: disable=arguments-differ
        del batch
        self.epochs += 1

    x, y = np.ones((10, 1)), np.ones((10, 1))
    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile('sgd', 'mse')

    my_cb = MyCallbackWithBatchHooks()
    cb_list = keras.callbacks.CallbackList([my_cb], verbose=0)
    self.assertTrue(cb_list._should_call_train_batch_hooks)  # pylint: disable=protected-access
    self.assertTrue(cb_list._should_call_test_batch_hooks)  # pylint: disable=protected-access
    self.assertTrue(cb_list._should_call_predict_batch_hooks)  # pylint: disable=protected-access

    model.fit(x, y, epochs=2, batch_size=10, callbacks=[my_cb], verbose=0)
    model.evaluate(x, y, batch_size=10, callbacks=[my_cb], verbose=0)
    model.predict(x, batch_size=10, callbacks=[my_cb], verbose=0)

    self.assertEqual(my_cb.train_batches, 2)
    self.assertEqual(my_cb.test_batches, 1)
    self.assertEqual(my_cb.predict_batches, 1)

    my_cb = MyCallbackWithoutBatchHooks()
    cb_list = keras.callbacks.CallbackList([my_cb], verbose=0)
    self.assertLen(cb_list.callbacks, 1)
    self.assertFalse(cb_list._should_call_train_batch_hooks)  # pylint: disable=protected-access
    self.assertFalse(cb_list._should_call_test_batch_hooks)  # pylint: disable=protected-access
    self.assertFalse(cb_list._should_call_predict_batch_hooks)  # pylint: disable=protected-access

    model.fit(x, y, epochs=2, batch_size=10, callbacks=[my_cb], verbose=0)
    model.evaluate(x, y, batch_size=10, callbacks=[my_cb], verbose=0)
    model.predict(x, batch_size=10, callbacks=[my_cb], verbose=0)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_implements_batch_hooks_override(self):
    class MyCallback(keras.callbacks.Callback):
      def __init__(self, should_run=True):  # pylint: disable=super-init-not-called
        self.should_run = should_run
        self.train_batches = 0
        self.test_batches = 0
        self.predict_batches = 0

      def on_train_batch_end(self, batch, logs=None):
        self.train_batches += 1

      def on_test_batch_end(self, batch, logs=None):
        self.test_batches += 1

      def on_predict_batch_end(self, batch, logs=None):
        self.predict_batches += 1

      def _implements_train_batch_hooks(self):
        return self.should_run

      def _implements_test_batch_hooks(self):
        return self.should_run

      def _implements_predict_batch_hooks(self):
        return self.should_run

    x, y = np.ones((10, 1)), np.ones((10, 1))
    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile('sgd', 'mse')

    my_cb = MyCallback(should_run=True)
    cb_list = keras.callbacks.CallbackList([my_cb], verbose=0)
    self.assertTrue(cb_list._should_call_train_batch_hooks)  # pylint: disable=protected-access
    self.assertTrue(cb_list._should_call_test_batch_hooks)  # pylint: disable=protected-access
    self.assertTrue(cb_list._should_call_predict_batch_hooks)  # pylint: disable=protected-access

    model.fit(x, y, epochs=2, batch_size=10, callbacks=[my_cb], verbose=0)
    model.evaluate(x, y, batch_size=10, callbacks=[my_cb], verbose=0)
    model.predict(x, batch_size=10, callbacks=[my_cb], verbose=0)

    self.assertEqual(my_cb.train_batches, 2)
    self.assertEqual(my_cb.test_batches, 1)
    self.assertEqual(my_cb.predict_batches, 1)

    my_cb = MyCallback(should_run=False)
    cb_list = keras.callbacks.CallbackList([my_cb], verbose=0)
    self.assertFalse(cb_list._should_call_train_batch_hooks)  # pylint: disable=protected-access
    self.assertFalse(cb_list._should_call_test_batch_hooks)  # pylint: disable=protected-access
    self.assertFalse(cb_list._should_call_predict_batch_hooks)  # pylint: disable=protected-access

    model.fit(x, y, epochs=2, batch_size=10, callbacks=[my_cb], verbose=0)
    model.evaluate(x, y, batch_size=10, callbacks=[my_cb], verbose=0)
    model.predict(x, batch_size=10, callbacks=[my_cb], verbose=0)

    self.assertEqual(my_cb.train_batches, 0)
    self.assertEqual(my_cb.test_batches, 0)
    self.assertEqual(my_cb.predict_batches, 0)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_default_callbacks_do_not_call_batch_hooks(self):
    model = keras.Sequential([keras.layers.Dense(1)])
    log_dir = self.get_temp_dir()
    cb_list = keras.callbacks.CallbackList([
        keras.callbacks.TensorBoard(log_dir, profile_batch=0),
        keras.callbacks.ModelCheckpoint(log_dir),
    ],
                                           add_progbar=True,
                                           model=model,
                                           verbose=2,
                                           epochs=3)
    self.assertLen(cb_list.callbacks, 3)
    self.assertFalse(cb_list._should_call_train_batch_hooks)  # pylint: disable=protected-access
    self.assertFalse(cb_list._should_call_test_batch_hooks)  # pylint: disable=protected-access
    self.assertFalse(cb_list._should_call_predict_batch_hooks)  # pylint: disable=protected-access

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_stop_training_batch_level(self):
    class MyCallback(keras.callbacks.Callback):
      def __init__(self):
        super(MyCallback, self).__init__()
        self.batch_counter = 0

      def on_train_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        if batch == 2:
          self.model.stop_training = True

    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile('sgd', 'mse')
    x, y = np.ones((10, 10)), np.ones((10, 1))
    my_cb = MyCallback()
    # Will run 5 batches if `stop_training` doesn't work.
    model.fit(x, y, batch_size=2, callbacks=[my_cb])
    self.assertEqual(my_cb.batch_counter, 3)


if __name__ == '__main__':
  test.main()
