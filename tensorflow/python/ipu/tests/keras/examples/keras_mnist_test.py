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
"""Comparitive test of upstream Keras vs IPU Keras on MNIST."""

from tensorflow.python import cast
from tensorflow.python import float32
from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.data import Dataset
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

# As the learning task in this test is less trivial than the typical
# consistency tests in keras_model_test.py (and related), in that it
# is a task on a real dataset that is larger, there is a small degree of
# output difference between IPU and CPU for the same initialisation.
#
# This is potentially due to cumulative numerical/precision differences
# between the two platforms? However, this test will fail if there is any
# significant loss difference between the the two platforms.
_tolerance = 1e-3


def create_dataset():
  mnist = keras.datasets.mnist

  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = x_train / 255.0

  train_ds = Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)
  train_ds = train_ds.map(lambda d, l: (cast(d, float32), cast(l, float32)))

  return train_ds.repeat()


def create_sequential_model(use_ipu=True):
  if use_ipu:
    Sequential = ipu.keras.Sequential
  else:
    Sequential = keras.Sequential

  init = keras.initializers.Constant(0.01)

  return Sequential([
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation='relu', kernel_initializer=init),
      keras.layers.Dense(10, activation='softmax', kernel_initializer=init)
  ])


def create_model(input_layer, use_ipu=True):
  if use_ipu:
    Model = ipu.keras.Model
  else:
    Model = keras.Model

  init = keras.initializers.Constant(0.01)

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
    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.auto_select_ipus(cfg, 1)
    ipu.utils.configure_ipu_system(cfg)

    ds = create_dataset()

    # Get IPU loss.
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      model = create_sequential_model()

      model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                    optimizer=keras.optimizer_v2.gradient_descent.SGD())

      ipu_out = model.fit(ds, steps_per_epoch=2000, epochs=4)

    # Get CPU loss.
    model = create_sequential_model(False)

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=keras.optimizer_v2.gradient_descent.SGD())

    cpu_out = model.fit(ds, steps_per_epoch=2000, epochs=4)

    # Verify.
    self.assertAllClose(ipu_out.history['loss'],
                        cpu_out.history['loss'],
                        rtol=_tolerance,
                        atol=_tolerance)

  @test_util.run_v2_only
  def testModelTraining(self):
    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.auto_select_ipus(cfg, 1)
    ipu.utils.configure_ipu_system(cfg)

    ds = create_dataset()

    # Get IPU loss.
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input((28, 28))
      model = create_model(input_layer)

      model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                    optimizer=keras.optimizer_v2.gradient_descent.SGD())

      ipu_out = model.fit(ds, steps_per_epoch=2000, epochs=4)

    # Get CPU loss.
    input_layer = keras.layers.Input((28, 28))
    model = create_model(input_layer, False)

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=keras.optimizer_v2.gradient_descent.SGD())

    cpu_out = model.fit(ds, steps_per_epoch=2000, epochs=4)

    # Verify.
    self.assertAllClose(ipu_out.history['loss'],
                        cpu_out.history['loss'],
                        rtol=_tolerance,
                        atol=_tolerance)


if __name__ == '__main__':
  test.main()
