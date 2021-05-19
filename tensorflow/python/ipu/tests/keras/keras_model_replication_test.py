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
from tensorflow.python.ipu.config import IPUConfig

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import keras
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import keras as ipu_keras


class IPUModelReplicatedTest(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=2)
  @test_util.run_v2_only
  def testPredictWithNumpyDataBs2Replicas2(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()

    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32),
                                       dtype=np.single,
                                       batch_size=2)
      init = keras.initializers.Constant(0.1)
      x = keras.layers.Dense(4, name="layer0",
                             kernel_initializer=init)(input_layer)
      x = keras.layers.Dense(2, name="layer1", kernel_initializer=init)(x)
      m = ipu_keras.Model(input_layer, x, gradient_accumulation_count=12)

      m.compile('sgd', loss='mse')

      # Input data
      input_x = np.full([96, 32], 1.0, dtype=np.single)

      # Generate predictions
      result = m.predict(input_x, batch_size=2)

      # The result is a Numpy array of predictions
      self.assertEqual(type(result), np.ndarray)
      self.assertEqual(result.shape, (96, 2))
      for i, r in enumerate(result):
        self.assertEqual(0, np.sum(r != result[i - 1]))

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.run_v2_only
  def testPredictWithNumpyDataBs2Replicas2Ga3DropsSamples(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()

    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32),
                                       dtype=np.single,
                                       batch_size=2)
      init = keras.initializers.Constant(0.1)
      x = keras.layers.Dense(4, name="layer0",
                             kernel_initializer=init)(input_layer)
      x = keras.layers.Dense(2, name="layer1", kernel_initializer=init)(x)
      m = ipu_keras.Model(input_layer, x, gradient_accumulation_count=3)

      m.compile('sgd', loss='mse')

      # Input data
      input_x = np.full([33, 32], 1.0, dtype=np.single)

      # with batch_size=2 this will give a dataset containing
      # 16 mini-batches
      # 16 // (3 * 2) = 2 steps per epoch per replica
      # dropping some samples

      # Generate predictions.
      # Ideally we would check the warning message here.
      # We just check that the output shape corresponds to the
      # expected value of 24 = (2 * 3 * 2 * 2)
      # (steps per epoch per replica * GA * RF * BS)
      result = m.predict(input_x, batch_size=2)
      self.assertEqual(result.shape[0], 24)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.run_v2_only
  def testPredictWithNumpyDataBs2Replicas2Steps2Ga1(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()

    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32),
                                       dtype=np.single,
                                       batch_size=2)
      init = keras.initializers.Constant(0.1)
      x = keras.layers.Dense(4, name="layer0",
                             kernel_initializer=init)(input_layer)
      x = keras.layers.Dense(2, name="layer1", kernel_initializer=init)(x)
      m = ipu_keras.Model(input_layer, x, gradient_accumulation_count=1)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Input data
      input_x = np.full([96, 32], 1.0, dtype=np.single)

      # Generate predictions
      result = m.predict(input_x, batch_size=2, steps=2)

      # The result is a Numpy array of predictions
      self.assertEqual(type(result), np.ndarray)
      self.assertEqual(result.shape, (4, 2))
      for i, r in enumerate(result):
        self.assertEqual(0, np.sum(r != result[i - 1]))

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.run_v2_only
  def testPredictMultipleOutputReplicas2(self):
    def predict_input_fn():
      x1 = np.ones((64, 32), dtype=np.float32)
      x2 = np.ones((64, 32), dtype=np.float32)
      x3 = np.ones((64, 32), dtype=np.float32)

      return (x1, x2, x3)

    # Intentional skip from input to middle of model.
    def model_fn():
      input_1 = keras.Input(32)
      input_2 = keras.Input(32)
      input_3 = keras.Input(32)

      init = keras.initializers.Constant(1)

      dense_1 = keras.layers.Dense(16,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu)(input_1)
      dense_2 = keras.layers.Dense(16,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu)(input_2)

      cat = keras.layers.Concatenate()([dense_1, dense_2, input_3])

      dense_3 = keras.layers.Dense(1,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu)(cat)
      dense_4 = keras.layers.Dense(1,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu)(cat)

      return ((input_1, input_2, input_3), (dense_3, dense_4))

    # IPU Test.
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      model = ipu_keras.Model(*model_fn())
      model.compile('sgd', ['mse', 'mse'])  #, loss_weights=[1.0, 0.2])

      predict_out = model.predict(predict_input_fn(), batch_size=4)

    # CPU Test.
    cpu_model = keras.Model(*model_fn())
    cpu_model.compile('sgd', ['mse', 'mse'])  #, loss_weights=[1.0, 0.2])

    cpu_predict_out = cpu_model.predict(predict_input_fn(), batch_size=4)

    # Comparison.
    self.assertEqual(np.shape(cpu_predict_out), np.shape(predict_out))

    for output in range(2):
      for cpu_predict, ipu_predict in zip(cpu_predict_out[output],
                                          predict_out[output]):
        np.testing.assert_almost_equal(cpu_predict, ipu_predict)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.run_v2_only
  def testPredictMultipleOutputDifferentShapesReplicas2(self):
    def predict_input_fn():
      x1 = np.ones((64, 32), dtype=np.float32)
      x2 = np.ones((64, 32), dtype=np.float32)
      x3 = np.ones((64, 32), dtype=np.float32)

      return (x1, x2, x3)

    # Intentional skip from input to middle of model.
    def model_fn():
      input_1 = keras.Input(32)
      input_2 = keras.Input(32)
      input_3 = keras.Input(32)

      init = keras.initializers.Constant(1)

      dense_1 = keras.layers.Dense(16,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu)(input_1)
      dense_2 = keras.layers.Dense(16,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu)(input_2)

      cat = keras.layers.Concatenate()([dense_1, dense_2, input_3])

      dense_3 = keras.layers.Dense(1,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu)(cat)
      dense_4 = keras.layers.Dense(2,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu)(cat)

      return ((input_1, input_2, input_3), (dense_3, dense_4))

    # CPU Test.
    cpu_model = keras.Model(*model_fn())
    cpu_model.compile('sgd', ['mse', 'mse'])  #, loss_weights=[1.0, 0.2])

    cpu_predict_out = cpu_model.predict(predict_input_fn(), batch_size=4)

    # IPU Test.
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      model = ipu_keras.Model(*model_fn())
      model.compile('sgd', ['mse', 'mse'])  #, loss_weights=[1.0, 0.2])

      predict_out = model.predict(predict_input_fn(), batch_size=4)

    # Comparison.
    self.assertEqual(np.shape(cpu_predict_out[0]), np.shape(predict_out[0]))
    for cpu_predict, ipu_predict in zip(cpu_predict_out[0], predict_out[0]):
      np.testing.assert_almost_equal(cpu_predict, ipu_predict)

    self.assertEqual(np.shape(cpu_predict_out[1]), np.shape(predict_out[1]))
    for cpu_predict, ipu_predict in zip(cpu_predict_out[1], predict_out[1]):
      np.testing.assert_almost_equal(cpu_predict[0], ipu_predict[0])
      np.testing.assert_almost_equal(cpu_predict[1], ipu_predict[1])


if __name__ == "__main__":
  googletest.main()
