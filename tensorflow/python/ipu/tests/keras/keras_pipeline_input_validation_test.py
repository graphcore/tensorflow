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

import numpy as np

from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


def simple_pipeline(x, layer_sizes, layer_stages, w=None):
  assert layer_sizes
  assert len(layer_sizes) == len(layer_stages)

  init = 'glorot_uniform'
  if w:
    assert w > 0
    init = keras.initializers.Constant(w)

  with ipu.keras.PipelineStage(layer_stages[0]):
    y = keras.layers.Dense(layer_sizes[0],
                           activation=keras.activations.relu,
                           kernel_initializer=init)(x)

  for n, stage in zip(layer_sizes[1:], layer_stages[1:]):
    with ipu.keras.PipelineStage(stage):
      y = keras.layers.Dense(n,
                             activation=keras.activations.relu,
                             kernel_initializer=init)(y)
  return y


def test_dataset(length=None, batch_size=1, x_val=1.0, y_val=0.2):
  constant_d = constant_op.constant(x_val, shape=[32])
  constant_l = constant_op.constant(y_val, shape=[2])

  ds = dataset_ops.Dataset.from_tensors((constant_d, constant_l))
  ds = ds.repeat(length)
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


def test_inference_dataset(length=None, batch_size=1, x_val=1.0):
  constant_d = constant_op.constant(x_val, shape=[32])

  ds = dataset_ops.Dataset.from_tensors(constant_d)
  ds = ds.repeat(length)
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


class IPUPipelineInputValidationTest(test.TestCase):
  @test_util.run_v2_only
  def testBadStage(self):
    with self.assertRaisesRegex(ValueError, "is not a valid pipeline stage"):
      input_layer = keras.layers.Input(shape=(32))
      _ = simple_pipeline(input_layer, [2, 4], [0, -1])

  @test_util.run_v2_only
  def testMissingStage(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 4], [0, 1])
      x = keras.layers.Dense(4)(x)

      with self.assertRaisesRegex(ValueError,
                                  "must have an associated pipeline stage"):
        ipu.keras.PipelineModel(inputs=input_layer,
                                outputs=x,
                                gradient_accumulation_count=2,
                                device_mapping=[0, 0])

  @test_util.run_v2_only
  def testNotEnoughIPUs(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 4, 2], [0, 1, 2])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=2)

      with self.assertRaisesRegex(ValueError,
                                  "Current device has 1 IPUs attached"):
        m.replication_factor  # pylint: disable=pointless-statement

  @test_util.run_v2_only
  def testBadStageOrder(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 4, 2], [0, 1, 0])

      with self.assertRaisesRegex(
          ValueError, "Layer dense_2 in pipeline stage 0 has a dependency"):
        ipu.keras.PipelineModel(inputs=input_layer,
                                outputs=x,
                                gradient_accumulation_count=2,
                                device_mapping=[0, 0])

  @test_util.run_v2_only
  def testCannotCallEagerly(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 4], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=4,
                                  device_mapping=[0, 0])

      c = constant_op.constant(np.zeros([1, 12], dtype=np.float32))

      with self.assertRaisesRegex(
          ValueError, "PipelineModel can only be called through the"):
        m(c)

  @test_util.run_v2_only
  def testCannotUseKerasV1Optimizers(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 4], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=4,
                                  device_mapping=[0, 0])

      with self.assertRaisesRegex(
          ValueError, "Optimizer must be a native TensorFlow or Keras V2"):
        opt = keras.optimizers.SGD(lr=0.001)
        m.compile(opt, 'mse')

  @test_util.run_v2_only
  def testMustCallCompileFit(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 2], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24,
                                  device_mapping=[0, 0])

      with self.assertRaisesRegex(
          RuntimeError, "You must compile your model before training/testing"):
        m.fit(test_dataset(length=64))

  @test_util.run_v2_only
  def testMustCallCompileEvaluate(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 2], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24,
                                  device_mapping=[0, 0])

      with self.assertRaisesRegex(
          RuntimeError, "You must compile your model before training/testing"):
        m.evaluate(test_dataset(length=64))

  @test_util.run_v2_only
  def testNeedTupleDatasetFit(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 2], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24,
                                  device_mapping=[0, 0])

      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError, r"requires a dataset with a structure containing "):
        m.fit(test_inference_dataset(length=48))

  @test_util.run_v2_only
  def testNeedTupleDatasetEvaluate(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 2], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24,
                                  device_mapping=[0, 0])

      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError, r"requires a dataset with a structure containing "):
        m.evaluate(test_inference_dataset(length=48))

  @test_util.run_v2_only
  def testNeedNonTupleDatasetPredict(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 2], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24,
                                  device_mapping=[0, 0])

      with self.assertRaisesRegex(
          ValueError, r"requires a dataset with a structure containing "):
        m.predict(test_dataset(length=48))

  @test_util.run_v2_only
  def testMismatchDatasetLengthToGradientAccumulationCount(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24,
                                  device_mapping=[0, 0])

      m.compile('sgd', loss='mse')
      with self.assertRaisesRegex(
          ValueError,
          "PipelineModel requires the number of mini-batches in the dataset "):
        m.fit(test_dataset(length=64), epochs=4)

  @test_util.run_v2_only
  def testUnlimitedDatasetHasNoStepsPerEpoch(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24,
                                  device_mapping=[0, 0])
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError, "When using an infinitely repeating dataset, you"):
        m.fit(test_dataset(), epochs=4)

  @test_util.run_v2_only
  def testStepsPerEpochTooLargeForDataset(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=12,
                                  device_mapping=[0, 0])

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      with self.assertRaisesRegex(
          ValueError,
          r"Steps per epoch times gradient accumulation count \(14 x 12\) is"
          r" greater than"):
        m.fit(test_dataset(length=144), steps_per_epoch=14)


if __name__ == '__main__':
  test.main()
