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

from test_optimizer import KernelLoggingOptimizer, AssertAllWeightsReplicaIdentical

from tensorflow import keras
from tensorflow.python import ipu
from tensorflow.python.platform import googletest
from tensorflow.python.ops import math_ops

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.python.framework import test_util

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers.experimental import preprocessing


def make_mnist_dataset(y_as_categorical=False):
  mnist = keras.datasets.mnist

  (x_train, y_train), _ = mnist.load_data()
  x_train = x_train / 255.0
  x_train = x_train[..., np.newaxis]

  x_train = x_train.astype('float32')
  y_train = to_categorical(y_train) if y_as_categorical else y_train.astype(
      'float32')
  train_ds = DatasetV2.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)

  return train_ds.repeat()


def make_imdb_dataset(max_features, minibatch_size):
  (x_train,
   y_train), (_, _) = keras.datasets.imdb.load_data(num_words=max_features)
  x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=80)

  dataset = DatasetV2.from_tensor_slices((x_train, y_train))
  dataset = dataset.repeat()
  dataset = dataset.map(lambda x, y: (x, math_ops.cast(y, np.int32)))
  return dataset.batch(minibatch_size, drop_remainder=True)


# This test is intended to verify that we get the same weight values produced on each replica
# when running simple models with the experimental.enable_prng_stability flag enabled. These tests
# are for TF2 only.
class PrngStabilityModelsKerasTest(test_util.TensorFlowTestCase):
  def setUp(self):
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 4
    cfg.floating_point_behaviour.esr = True
    cfg.experimental.enable_prng_stability = True
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    keras.backend.set_floatx('float16')

    self.outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

  def createLoggingOptimizer(self, wrapped_optimizer, model):
    return KernelLoggingOptimizer(self.outfeed, wrapped_optimizer, model)

  def assertAllWeightsReplicaIdentical(self, expected_replicas=2):
    out_queue = self.outfeed.dequeue()
    self.assertTrue(out_queue, "Expected some logged variables.")

    for var in out_queue:
      var = var.numpy()
      AssertAllWeightsReplicaIdentical(self, var, expected_replicas)

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.run_v2_only
  def testLinearRegression(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      np.random.seed(1234)
      horsepower = np.random.rand(320, 1).astype(np.float32)
      mpg = np.random.rand(320, 1).astype(np.float32)

      normalizer = preprocessing.Normalization(input_shape=[
          1,
      ], axis=None)
      normalizer.adapt(horsepower)

      model = keras.Sequential([normalizer, keras.layers.Dense(units=1)])
      model.set_pipelining_options(gradient_accumulation_steps_per_replica=4)
      model.set_pipeline_stage_assignment([0, 1])

      model.compile(optimizer=self.createLoggingOptimizer(
          keras.optimizers.Adam(learning_rate=0.1), model),
                    loss='mean_absolute_error',
                    steps_per_execution=64)
      model.fit(horsepower, mpg, epochs=3, steps_per_epoch=64)

      self.assertAllWeightsReplicaIdentical()

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.run_v2_only
  def testImdbRnn(self):
    max_features = 10000
    minibatch_size = 32

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      dataset = make_imdb_dataset(max_features, minibatch_size)

      model = keras.Sequential([
          keras.layers.Input(shape=(80),
                             dtype=np.int32,
                             batch_size=minibatch_size),
          keras.layers.Embedding(max_features, 128),
          keras.layers.Bidirectional(keras.layers.LSTM(64)),
          keras.layers.Dense(64, activation='relu'),
          keras.layers.Dense(1)
      ])
      model.set_pipelining_options(gradient_accumulation_steps_per_replica=16)
      model.set_pipeline_stage_assignment([0, 0, 1, 1])

      model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                    optimizer=self.createLoggingOptimizer(
                        keras.optimizers.Adam(1e-4), model),
                    steps_per_execution=64)
      model.fit(dataset, steps_per_epoch=64, epochs=3)

      self.assertAllWeightsReplicaIdentical()

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.run_v2_only
  def testImdb(self):
    max_features = 20000
    minibatch_size = 32
    gradient_accumulation_steps_per_replica = 16

    def get_model():
      input_layer = keras.layers.Input(shape=(80),
                                       dtype=np.int32,
                                       batch_size=minibatch_size)
      with ipu.keras.PipelineStage(0):
        x = keras.layers.Embedding(max_features, 128)(input_layer)

      with ipu.keras.PipelineStage(1):
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(1)(x)

      return keras.Model(input_layer, x)

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      dataset = make_imdb_dataset(max_features, minibatch_size)

      model = get_model()
      model.set_pipelining_options(gradient_accumulation_steps_per_replica=
                                   gradient_accumulation_steps_per_replica)

      model.compile(steps_per_execution=384,
                    loss='binary_crossentropy',
                    optimizer=self.createLoggingOptimizer(
                        keras.optimizers.Adam(0.005), model))
      model.fit(dataset, steps_per_epoch=768, epochs=2)

      self.assertAllWeightsReplicaIdentical()

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.run_v2_only
  def testMnistCnn(self):
    def create_model():
      return keras.Sequential([
          keras.layers.Conv2D(64,
                              kernel_size=3,
                              activation="relu",
                              input_shape=(28, 28, 1)),
          keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
          keras.layers.Flatten(),
          keras.layers.Dense(10, activation='softmax')
      ])

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      dataset = make_mnist_dataset(y_as_categorical=True)

      steps_per_epoch = 4
      model = create_model()
      model.compile(loss='categorical_crossentropy',
                    optimizer=self.createLoggingOptimizer(
                        keras.optimizers.SGD(), model),
                    steps_per_execution=steps_per_epoch)
      model.fit(dataset, epochs=3, steps_per_epoch=steps_per_epoch)

      self.assertAllWeightsReplicaIdentical(expected_replicas=4)

  @test_util.run_v2_only
  @tu.test_uses_ipus(num_ipus=4)
  def testMnist(self):
    def create_model():
      return keras.Sequential([
          keras.layers.Flatten(),
          keras.layers.Dense(128, activation='relu'),
          keras.layers.Dense(10, activation='softmax')
      ])

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      dataset = make_mnist_dataset()

      model = create_model()
      model.set_pipelining_options(gradient_accumulation_steps_per_replica=4)
      model.set_pipeline_stage_assignment([0, 1, 1])

      steps_per_epoch = 16
      model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                    optimizer=self.createLoggingOptimizer(
                        keras.optimizers.SGD(), model),
                    steps_per_execution=steps_per_epoch)
      model.fit(dataset, epochs=3, steps_per_epoch=steps_per_epoch)

      self.assertAllWeightsReplicaIdentical()


if __name__ == "__main__":
  googletest.main()
