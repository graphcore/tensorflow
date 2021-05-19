# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Comparative test of upstream Keras vs IPU Keras on MNIST."""

import numpy as np
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python import cast
from tensorflow.python import float32
from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.data import Dataset
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

# As the learning task in these tests is less trivial than the typical
# consistency tests in keras_model_test.py (and related), in that it
# is a task on a real dataset that is larger, there is a small degree of
# output difference between IPU and CPU for the same initialisation.
#
# This is potentially due to cumulative numerical/precision differences
# between the two platforms? However, the training tests will fail if there is any
# significant loss difference between the the two platforms.
_tolerance = 1e-3

# To compare predictions, we start from a random initialisation in order
# to train the models to a higher accuracy in 4 epochs
# We expect less than 5% of the predictions of the most-likely digit to differ
_prediction_tolerance = 0.05


def create_datasets():
  mnist = keras.datasets.mnist

  (x_train, y_train), (x_test, _) = mnist.load_data()
  x_train = x_train / 255.0
  x_test = x_test / 255.0

  train_ds = Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)
  train_ds = train_ds.map(lambda d, l: (cast(d, float32), cast(l, float32)))

  x_test = x_test.astype('float32')
  test_ds = Dataset.from_tensor_slices(x_test).batch(32, drop_remainder=True)

  return train_ds.repeat(), test_ds


def create_sequential_model(use_ipu=True, constant_init=True):
  if use_ipu:
    Sequential = ipu.keras.Sequential
  else:
    Sequential = keras.Sequential

  if constant_init:
    init = keras.initializers.Constant(0.01)
  else:
    init = None

  return Sequential([
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation='relu', kernel_initializer=init),
      keras.layers.Dense(10, activation='softmax', kernel_initializer=init)
  ])


def create_model(input_layer, use_ipu=True, constant_init=True):
  if use_ipu:
    Model = ipu.keras.Model
  else:
    Model = keras.Model

  if constant_init:
    init = keras.initializers.Constant(0.01)
  else:
    init = None

  x = keras.layers.Flatten()(input_layer)
  x = keras.layers.Dense(128, activation='relu', kernel_initializer=init)(x)
  x = keras.layers.Dense(10, activation='softmax', kernel_initializer=init)(x)

  return Model(input_layer, x)


class IPUKerasMNISTTest(test.TestCase):
  """
  This test is taken from the example failure case of ticket
  https://phabricator.sourcevertex.net/T30486
  """
  @test_util.run_v2_only
  def testSequentialTraining(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    train_ds, _ = create_datasets()

    # Get IPU loss.
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      model = create_sequential_model()

      model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                    optimizer=keras.optimizer_v2.gradient_descent.SGD())

      ipu_out = model.fit(train_ds, steps_per_epoch=2000, epochs=4)

    # Get CPU loss.
    model = create_sequential_model(False)

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=keras.optimizer_v2.gradient_descent.SGD())

    cpu_out = model.fit(train_ds, steps_per_epoch=2000, epochs=4)

    # Verify.
    self.assertAllClose(ipu_out.history['loss'],
                        cpu_out.history['loss'],
                        rtol=_tolerance,
                        atol=_tolerance)

  @test_util.run_v2_only
  def testSequentialPredictions(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    train_ds, test_ds = create_datasets()

    # Train IPU model and get predictions
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      model = create_sequential_model(constant_init=False)

      model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                    optimizer=keras.optimizer_v2.gradient_descent.SGD())

      model.fit(train_ds, steps_per_epoch=2000, epochs=4)

      ipu_predictions = model.predict(test_ds, steps=20)

    # Train CPU model and get predictions
    model = create_sequential_model(False, constant_init=False)

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=keras.optimizer_v2.gradient_descent.SGD())

    model.fit(train_ds, steps_per_epoch=2000, epochs=4)

    cpu_predictions = model.predict(test_ds, steps=20)

    # compare predictions: expect fewer than _prediction_tolerance of the
    # most-likely digits to differ
    self.assertEqual(cpu_predictions.shape, ipu_predictions.shape)
    num_different = np.sum(
        np.argmax(cpu_predictions, axis=1) != np.argmax(ipu_predictions,
                                                        axis=1))
    self.assertLess(num_different / len(cpu_predictions),
                    _prediction_tolerance)

  @test_util.run_v2_only
  def testModelTraining(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    train_ds, _ = create_datasets()

    # Get IPU loss.
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input((28, 28))
      model = create_model(input_layer)

      model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                    optimizer=keras.optimizer_v2.gradient_descent.SGD())

      ipu_out = model.fit(train_ds, steps_per_epoch=2000, epochs=4)

    # Get CPU loss.
    input_layer = keras.layers.Input((28, 28))
    model = create_model(input_layer, False)

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=keras.optimizer_v2.gradient_descent.SGD())

    cpu_out = model.fit(train_ds, steps_per_epoch=2000, epochs=4)

    # Verify.
    self.assertAllClose(ipu_out.history['loss'],
                        cpu_out.history['loss'],
                        rtol=_tolerance,
                        atol=_tolerance)

  @test_util.run_v2_only
  def testModelPredictions(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    train_ds, test_ds = create_datasets()

    # Train IPU model and get predictions
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input((28, 28))
      model = create_model(input_layer, constant_init=False)

      model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                    optimizer=keras.optimizer_v2.gradient_descent.SGD())

      model.fit(train_ds, steps_per_epoch=2000, epochs=4)

      ipu_predictions = model.predict(test_ds, steps=20)

    # Train CPU model and get predictions
    input_layer = keras.layers.Input((28, 28))
    model = create_model(input_layer, False, constant_init=False)

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=keras.optimizer_v2.gradient_descent.SGD())

    model.fit(train_ds, steps_per_epoch=2000, epochs=4)

    cpu_predictions = model.predict(test_ds, steps=20)

    # compare predictions: expect fewer than _prediction_tolerance of the
    # most-likely digits to differ
    self.assertEqual(cpu_predictions.shape, ipu_predictions.shape)
    num_different = np.sum(
        np.argmax(cpu_predictions, axis=1) != np.argmax(ipu_predictions,
                                                        axis=1))
    self.assertLess(num_different / len(cpu_predictions),
                    _prediction_tolerance)


if __name__ == '__main__':
  test.main()
