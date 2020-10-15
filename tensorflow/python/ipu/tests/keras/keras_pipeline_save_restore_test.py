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
"""Test for IPU Keras Pipelined model save and restore."""

import os
import shutil

import numpy as np

from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


def test_dataset(length=None, batch_size=1):

  constant_d = constant_op.constant(1.0, shape=[32])
  constant_l = constant_op.constant(0.2, shape=[2])

  ds = dataset_ops.Dataset.from_tensors((constant_d, constant_l))
  ds = ds.repeat(length)
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


def fixed_weight_pipeline():
  return [
      [
          keras.layers.Dense(
              4,
              name="layer0",
              kernel_initializer=keras.initializers.Constant(0.1),
              bias_initializer=keras.initializers.Constant(0.0)),
      ],
      [
          keras.layers.Dense(
              2,
              name="layer1",
              kernel_initializer=keras.initializers.Constant(0.1),
              bias_initializer=keras.initializers.Constant(0.0)),
      ],
  ]


class IPUPipelineTest(test.TestCase):
  @test_util.run_v2_only
  def testCanSaveWeights(self):

    dataset = test_dataset(length=72)

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.auto_select_ipus(cfg, 2)
    ipu.utils.configure_ipu_system(cfg)

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu.keras.SequentialPipelineModel(fixed_weight_pipeline(),
                                            gradient_accumulation_count=24)
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      h = m.fit(dataset, epochs=1, verbose=0)
      loss = h.history['loss'][0]

    # Record the current weights
    weights = m.weights
    self.assertEqual(4, len(weights))

    # Save weights to file
    m.save_weights("trained_model_weights")

    # Reset the weights and check that they are different
    zeroed_weights = []
    for w in weights:
      zeroed_weights.append(np.zeros(w.shape))
    m.set_weights(zeroed_weights)

    for w1, w2 in zip(m.weights, zeroed_weights):
      self.assertTrue(np.all(w1.numpy() == np.array(w2)))

    # Restore the weights and check that they are back to the trained
    with strategy.scope():
      m = ipu.keras.SequentialPipelineModel(fixed_weight_pipeline(),
                                            gradient_accumulation_count=24)
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Can restore weights
      m.load_weights("trained_model_weights")

      # Make sure we can continue training
      h = m.fit(dataset, epochs=1, verbose=0)

      # New loss is not the same as last loss (as it would be if starting
      # from the same initial values)
      self.assertNotEqual(h.history['loss'][0], loss)

      # Weights were restored and have been further trained
      self.assertEqual(4, len(m.weights))
      for w1, w2 in zip(m.weights, weights):
        self.assertTrue(np.any(w1.numpy() != np.array(w2)))

  @test_util.run_v2_only
  def testCanDoTrainingCheckpoint(self):

    dataset = test_dataset(length=72)

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.auto_select_ipus(cfg, 2)
    ipu.utils.configure_ipu_system(cfg)

    checkpoint_path = "ckpt/model"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Remove directory from any previous run
    if os.path.isdir(checkpoint_dir):
      shutil.rmtree(checkpoint_dir)

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu.keras.SequentialPipelineModel(fixed_weight_pipeline(),
                                            gradient_accumulation_count=24)
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

      h = m.fit(dataset, callbacks=[cp_callback], verbose=0, epochs=1)
      loss = h.history['loss'][0]

    # Has it created a file
    self.assertTrue(os.path.isdir(checkpoint_dir))

    with strategy.scope():

      # Create a new model and train (should restore from checkpoint)
      m = ipu.keras.SequentialPipelineModel(fixed_weight_pipeline(),
                                            gradient_accumulation_count=24)
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Can restore weights
      m.load_weights(checkpoint_path)

      # Verify that the loss wasn't the same, because the weights were
      # restored from a checkoiint.
      h = m.fit(dataset, verbose=0, epochs=1)
      self.assertNotEqual(h.history['loss'][0], loss)

  @test_util.run_v2_only
  def testInvalidRestorePath(self):

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.auto_select_ipus(cfg, 2)
    ipu.utils.configure_ipu_system(cfg)

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu.keras.SequentialPipelineModel(fixed_weight_pipeline(),
                                            gradient_accumulation_count=24)
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      with self.assertRaisesRegex(errors.NotFoundError,
                                  r"Failed to find any matching files"):
        m.load_weights("random_bad_path")


if __name__ == '__main__':
  test.main()
