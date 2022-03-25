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
import tempfile
import numpy as np
from absl.testing import parameterized

from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


def get_mnist_dataset(batch_size):
  (x_train, _), (x_test, _) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0

  # Add a channels dimension.
  x_test = x_test[..., np.newaxis]
  x_test = x_test.astype('float32')

  predict_ds = dataset_ops.DatasetV2.from_tensor_slices(x_test).batch(
      batch_size, drop_remainder=True).repeat()

  return predict_ds


def simple_sequential_model():
  return keras.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(10,
                         activation='softmax',
                         kernel_initializer=keras.initializers.Constant(0.5),
                         bias_initializer='zeros')
  ])


def simple_functional_model():
  d = keras.layers.Input((28, 28))
  x = keras.layers.Flatten()(d)
  x = keras.layers.Dense(10,
                         activation='softmax',
                         kernel_initializer=keras.initializers.Constant(0.5),
                         bias_initializer='zeros')(x)
  return keras.Model(d, x)


class KerasFeedOptionsTest(test.TestCase, parameterized.TestCase):
  TESTCASES = [{
      "testcase_name": "sequential",
      "model_fn": simple_sequential_model,
  }, {
      "testcase_name": "functional",
      "model_fn": simple_functional_model,
  }]

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_v2_only
  def testOptions(self, model_fn):
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    batch_size = 2
    steps = 64
    ds = get_mnist_dataset(batch_size)

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      model = model_fn()
      model.compile(steps_per_execution=steps)
      model.set_infeed_queue_options(prefetch_depth=2)
      model.set_outfeed_queue_options(buffer_depth=2)

      # Run predict - check no compilation errors.
      model.predict(ds, steps=steps)

      with tempfile.TemporaryDirectory() as tmpdir:
        # Test saving model.
        model_file = tmpdir + "/model"
        model.save(model_file)

        restored_model = keras.models.load_model(model_file)
        self.assertEqual(restored_model._infeed_kwargs, {"prefetch_depth": 2})  # pylint: disable=protected-access
        self.assertEqual(restored_model._outfeed_kwargs, {"buffer_depth": 2})  # pylint: disable=protected-access


if __name__ == '__main__':
  test.main()
