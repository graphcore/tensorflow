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
from tempfile import TemporaryDirectory
from tensorflow.python.ipu.config import IPUConfig
import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.python import keras
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import keras as ipu_keras


class IPUModelReplicatedMnistTest(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=4)
  @test_util.run_v2_only
  def testCompareMnistPredictionsWithCpu(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    def model_fn():
      input_layer = keras.layers.Input(shape=(28, 28, 1),
                                       dtype='float32',
                                       batch_size=32)
      x = keras.layers.Flatten()(input_layer)
      x = keras.layers.Dense(128, activation='relu')(x)
      x = keras.layers.Dense(10, activation='softmax')(x)

      return input_layer, x

    def create_datasets():
      (x_train, y_train), (x_test, _) = mnist.load_data()
      x_train, x_test = x_train / 255.0, x_test / 255.0

      # Add a channels dimension.
      x_train = x_train[..., np.newaxis]
      x_test = x_test[..., np.newaxis]

      x_train = x_train.astype('float32')
      y_train = y_train.astype('float32')
      x_test = x_test.astype('float32')

      train_ds = DatasetV2.from_tensor_slices(
          (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)

      predict_ds = DatasetV2.from_tensor_slices(x_test[:312 * 32]).batch(
          32, drop_remainder=True)

      return train_ds.repeat(), predict_ds

    dataset, predict_ds = create_datasets()

    # train on CPU
    model = keras.Model(*model_fn())
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'],
                  optimizer=keras.optimizer_v2.gradient_descent.SGD(),
                  steps_per_execution=4)
    model.fit(dataset, epochs=2, steps_per_epoch=2000)

    # Predict on CPU.
    cpu_predictions = model.predict(predict_ds, steps=12)

    with TemporaryDirectory() as tmpdir:
      # Test saving weights.
      weights_file = tmpdir + "/weights.hd5"
      model.save_weights(weights_file)

      # Predict on IPU with replication.
      strategy = ipu_strategy.IPUStrategyV1()
      with strategy.scope():
        ipu_model = keras.Model(*model_fn())
        ipu_model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'],
                          optimizer=keras.optimizer_v2.gradient_descent.SGD(),
                          steps_per_execution=4)
        ipu_model.load_weights(weights_file).expect_partial()
        ipu_predictions = ipu_model.predict(predict_ds, steps=12)

      # Compare predictions.
      self.assertAllClose(cpu_predictions, ipu_predictions)

    with TemporaryDirectory() as tmpdir:
      # Test saving model.
      model_file = tmpdir + "/model"
      model.save(model_file)

      # Predict on IPU with replication
      strategy = ipu_strategy.IPUStrategyV1()
      with strategy.scope():
        ipu_model = keras.models.load_model(model_file)
        self.assertIsInstance(
            ipu_model,
            ipu_keras.extensions.keras_extension_base.KerasExtensionBase)
        ipu_model.compile(steps_per_execution=4)
        ipu_predictions = ipu_model.predict(predict_ds, steps=12)

      # compare predictions
      self.assertAllClose(cpu_predictions, ipu_predictions)


if __name__ == "__main__":
  googletest.main()
