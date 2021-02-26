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

from functools import partial
import re

import numpy as np

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.python import dtypes
from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ipu.tests import pipelining_test_util
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import vis_utils
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


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


def simple_sequential_pipeline(layer_sizes, layer_stages, w=None):
  assert layer_sizes
  assert len(layer_sizes) == len(layer_stages)

  init = 'glorot_uniform'
  if w:
    assert w > 0
    init = keras.initializers.Constant(w)

  stages = []
  prev_stage = -1
  for n, s in zip(layer_sizes, layer_stages):
    if not stages or s != prev_stage:
      stages.append([])
    stages[-1].append(
        keras.layers.Dense(n,
                           activation=keras.activations.relu,
                           kernel_initializer=init))
    prev_stage = s
  return stages


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


def test_language_dataset(length=None, batch_size=1):

  constant_d = constant_op.constant(1, shape=[32], dtype=np.int32)
  constant_l = constant_op.constant(2, shape=[32], dtype=np.int32)

  ds = dataset_ops.Dataset.from_tensors((constant_d, constant_l))
  ds = ds.repeat(length)
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


@def_function.function
def run_model_on_cpu(test_wrapper, model, input_values, repeat_count,
                     gradient_accumulation_count, loss, optimizer):
  def inputs_fn():
    return []

  def stage_fn(stage_id, x, t):
    assert stage_id < len(model)

    for l in model[stage_id]:
      x = l(x)

    if stage_id == len(model) - 1:
      if loss:
        return loss(y_true=t, y_pred=x)
      return x

    return x, t

  stages = []
  for i in range(len(model)):
    stages.append(partial(stage_fn, i))

  outputs = pipelining_test_util.PipelineTester.run_on_cpu(
      test_wrapper, stages, inputs_fn, input_values, repeat_count,
      gradient_accumulation_count, test_dataset, optimizer)
  return outputs


def aggregate_cpu_out(aggregator, results):
  a = aggregator(use_steps=False, num_samples=len(results))
  e = enumerate(iter(results))

  i, r = next(e)
  a.create(r)
  a.aggregate(r, 0, 1)

  for i, r in e:
    a.aggregate(r, i, i + 1)
  a.finalize()
  return a.results


class BatchCallbackCounter(keras.callbacks.Callback):
  def __init__(self):
    super(BatchCallbackCounter, self).__init__()
    self._count = 0
    self._logs = []

  def on_batch_end(self, batch, logs=None):
    self._logs.append(logs)
    self._count = self._count + 1

  def count(self):
    return self._count

  def logs(self):
    return self._logs


def _count_host_to_device_events(evts):
  evt_types = ipu.utils.extract_all_types_from_event_trace(evts)
  evt_types = filter(lambda x: x == IpuTraceEvent.HOST_TO_DEVICE_TRANSFER,
                     evt_types)
  return len(list(evt_types))


class IPUPipelineTest(test.TestCase):
  @test_util.run_v2_only
  def testBadStage(self):
    with self.assertRaisesRegex(ValueError, "is not a valid pipeline stage"):
      input_layer = keras.layers.Input(shape=(32))
      _ = simple_pipeline(input_layer, [2, 4], [0, -1])

  @test_util.run_v2_only
  def testMissingStage(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 4], [0, 1])
      x = keras.layers.Dense(4)(x)

      with self.assertRaisesRegex(ValueError,
                                  "must have an associated pipeline stage"):
        ipu.keras.PipelineModel(inputs=input_layer,
                                outputs=x,
                                gradient_accumulation_count=2)

  @test_util.run_v2_only
  def testNotEnoughIPUs(self):
    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.auto_select_ipus(cfg, 2)
    ipu.utils.configure_ipu_system(cfg)

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 4, 2], [0, 1, 2])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=2)

      with self.assertRaisesRegex(ValueError,
                                  "Current device has 2 IPUs attached"):
        m.replication_factor  # pylint: disable=pointless-statement

  @test_util.run_v2_only
  def testBadStageOrder(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 4, 2], [0, 1, 0])

      with self.assertRaisesRegex(
          ValueError, "Layer dense_2 in pipeline stage 0 has a dependency"):
        ipu.keras.PipelineModel(inputs=input_layer,
                                outputs=x,
                                gradient_accumulation_count=2)

  @test_util.run_v2_only
  def testCannotCallEagerly(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 4], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=4)

      c = constant_op.constant(np.zeros([1, 12], dtype=np.float32))

      with self.assertRaisesRegex(
          ValueError, "PipelineModel can only be called through the"):
        m(c)

  @test_util.run_v2_only
  def testCannotUseKerasV1Optimizers(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 4], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=4)

      with self.assertRaisesRegex(
          ValueError,
          "Optimizer must be a native Tensorflow optimizer, or a Keras V2"):
        opt = keras.optimizers.SGD(lr=0.001)
        m.compile(opt, 'mse')

  @test_util.run_v2_only
  def testMustCallCompileFit(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 2], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)

      with self.assertRaisesRegex(
          RuntimeError, "You must compile your model before training/testing"):
        m.fit(test_dataset(length=64))

  @test_util.run_v2_only
  def testMustCallCompileEvaluate(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 2], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)

      with self.assertRaisesRegex(
          RuntimeError, "You must compile your model before training/testing"):
        m.evaluate(test_dataset(length=64))

  @test_util.run_v2_only
  def testNeedTupleDatasetFit(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 2], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)

      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError, r"requires a dataset with a structure containing "):
        m.fit(test_inference_dataset(length=48))

  @test_util.run_v2_only
  def testNeedTupleDatasetEvaluate(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 2], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)

      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError, r"requires a dataset with a structure containing "):
        m.evaluate(test_inference_dataset(length=48))

  @test_util.run_v2_only
  def testNeedNonTupleDatasetPredict(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [2, 2], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)

      with self.assertRaisesRegex(
          ValueError, r"requires a dataset with a structure containing "):
        m.predict(test_dataset(length=48))

  @test_util.run_v2_only
  def testMismatchDatasetLengthToGradientAccumulationCount(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)

      m.compile('sgd', loss='mse')
      with self.assertRaisesRegex(
          ValueError,
          "PipelineModel requires the number of mini-batches in the dataset "):
        m.fit(test_dataset(length=64), epochs=4)

  @test_util.run_v2_only
  def testUnlimitedDatasetHasNoStepsPerEpoch(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError, "When using an infinitely repeating dataset, you"):
        m.fit(test_dataset(), epochs=4)

  @test_util.run_v2_only
  def testStepsPerEpochTooLargeForDataset(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=12)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      with self.assertRaisesRegex(
          ValueError,
          r"Steps per epoch times gradient accumulation count \(14 x 12\) is"
          r" greater than"):
        m.fit(test_dataset(length=144), steps_per_epoch=14)

  @test_util.run_v2_only
  def testResultsOneEpochWithTfOptimizer_CpuMatch(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1], w=0.2)
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=8)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = gradient_descent.GradientDescentOptimizer(0.001)

      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      history = m.fit(test_dataset(length=96))

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 1)
      self.assertEqual(type(history.history['loss'][0]), np.float64)

    opt_cpu = gradient_descent.GradientDescentOptimizer(0.001)
    loss_cpu = keras.losses.mean_squared_error
    cpu_loss = run_model_on_cpu(
        self, simple_sequential_pipeline([32, 2], [0, 1], w=0.2), [], 12, 8,
        loss_cpu, opt_cpu)
    cpu_loss = list(map(lambda x: x.numpy(), cpu_loss))
    cpu_loss = aggregate_cpu_out(training_utils.MetricsAggregator, cpu_loss)
    cpu_loss = cpu_loss[0]

    # history['loss'] is one loss value per epoch (of which there is 1)
    ipu_loss = history.history['loss'][0]

    self.assertAllClose(ipu_loss, cpu_loss, rtol=1e-5)

  @test_util.run_v2_only
  def testFitHistoryWithKerasOptimizer(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1], w=0.2)
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      history = m.fit(test_dataset(length=72))

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 1)
      self.assertEqual(type(history.history['loss'][0]), np.float64)

  @test_util.run_v2_only
  def testFitHistoryTwoEpochs(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1], w=0.2)
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      history = m.fit(test_dataset(length=72), epochs=2)

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 2)
      self.assertEqual(type(history.history['loss'][0]), np.float64)
      self.assertEqual(type(history.history['loss'][1]), np.float64)

  @test_util.run_v2_only
  def testFitHistoryStepsPerRun(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1], w=0.2)
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # With a strategy saying 2 steps per run, and a step having a
      # gradient_accumulation_count=24 mini-batches, we should consume a 96 sample
      # epoch in 2 runs.  So we expect 2 'per-batch' callbacks.
      cb = BatchCallbackCounter()

      # Fit the weights to the dataset
      m.fit(test_dataset(length=96), callbacks=[cb], steps_per_run=2)

      # Should have been called back twice due to end of batch
      self.assertEqual(cb.count(), 2)

  @test_util.run_v2_only
  def testFitHistoryStepsPerEpochOneEpoch(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1], w=0.2)
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      history = m.fit(test_dataset(), steps_per_epoch=144)

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 1)
      self.assertEqual(type(history.history['loss'][0]), np.float64)

  @test_util.run_v2_only
  def testFitTwice(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      ds = test_dataset()

      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 32, 2], [0, 1, 1])
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=8)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Clear profiling logs
      ipu.ops.summary_ops.get_ipu_reports()

      # Fit the weights to the dataset
      history = m.fit(ds, steps_per_epoch=2)
      l = history.history['loss'][0]

      # # Record weights
      w_1 = [w.numpy() for w in m.weights]

      # Fit the weights to the dataset
      history = m.fit(ds, steps_per_epoch=2)

      # Loss should be different after second training.
      self.assertTrue(l > history.history['loss'][0])

      w_2 = [w.numpy() for w in m.weights]

      # Weights should be different too.
      for w1, w2 in zip(w_1, w_2):
        self.assertFalse(np.all(w1 == w2))

      # Should have compiled the graph once, and executed twice.
      evts = ipu.ops.summary_ops.get_ipu_reports()
      evts = ipu.utils.extract_compile_reports(evts)
      self.assertEqual(1, len(evts))

      # Fit the weights with a new dataset
      history = m.fit(test_dataset(), steps_per_epoch=2)

      # Loss should be different after second training.
      self.assertTrue(l > history.history['loss'][0])

      w_3 = [w.numpy() for w in m.weights]

      # Weights should be different too.
      for w2, w3 in zip(w_2, w_3):
        self.assertFalse(np.all(w2 == w3))

      # Should have compiled the graph once more
      evts = ipu.ops.summary_ops.get_ipu_reports()
      evts = ipu.utils.extract_compile_reports(evts)
      self.assertEqual(1, len(evts))

  @test_util.run_v2_only
  def testFitHistoryStepsPerEpochTwoEpochs(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1], w=0.2)
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      history = m.fit(test_dataset(), steps_per_epoch=144, epochs=2)

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 2)
      self.assertEqual(type(history.history['loss'][0]), np.float64)
      self.assertEqual(type(history.history['loss'][1]), np.float64)

  @test_util.run_v2_only
  def testFitHistoryWithKerasOptimizerBatchSize2(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1], w=0.2)
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      cb = BatchCallbackCounter()
      m.fit(test_dataset(length=144, batch_size=2), callbacks=[cb])

      # 144 samples, batch size 2 = 72 mini-batches
      # 72 mini-batches, pipeline depth 24 = 3 steps
      logs = cb.logs()
      self.assertEqual(len(logs), 1)
      self.assertTrue("num_steps" in logs[0].keys())
      self.assertEqual(logs[0]['num_steps'], 3)

  @test_util.run_v2_only
  def testFitMultipleOutputs(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      init = keras.initializers.Constant(0.1)

      with ipu.keras.PipelineStage(0):
        y1 = keras.layers.Dense(2,
                                activation=keras.activations.relu,
                                kernel_initializer=init,
                                name="output1")(input_layer)

      with ipu.keras.PipelineStage(1):
        y2 = keras.layers.Dense(2, kernel_initializer=init,
                                name="output2")(input_layer)

      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=[y1, y2],
                                  gradient_accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      dataset = test_dataset(length=144, batch_size=2)

      def d(x, y):
        return x, y, y

      dataset = dataset.map(d)

      history = m.fit(dataset, epochs=2)
      self.assertEqual(set(history.history.keys()),
                       set(['loss', 'output1_loss', 'output2_loss']))
      self.assertEqual(type(history.history['loss']), list)
      losses = history.history['loss']
      self.assertEqual(len(losses), 2)
      self.assertTrue(losses[0] > losses[-1])

  @test_util.run_v2_only
  def testFitWithLearningRateDecay(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      # Clear old reports
      ipu.ops.summary_ops.get_ipu_reports()

      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1], w=0.2)
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001,
                                                    decay=0.1)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      m.fit(test_dataset(length=72), epochs=6)

      # Ensure that we are not downloading the weights each time even though
      # the 'learning rate' hyper is being updated
      evts = ipu.ops.summary_ops.get_ipu_reports()
      self.assertEqual(1, _count_host_to_device_events(evts))

  @test_util.run_v2_only
  def testFitWithExponentialDecayLearningRateSchedule(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      # Clear old reports
      ipu.ops.summary_ops.get_ipu_reports()

      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1], w=0.2)
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      lrs = keras.optimizer_v2.learning_rate_schedule.ExponentialDecay(
          0.001, 4, 0.1, staircase=True)
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=lrs)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      m.fit(test_dataset(length=72), epochs=1)

      # Ensure that we are not downloading the weights each time even though
      # the 'learning rate' hyper is being updated
      evts = ipu.ops.summary_ops.get_ipu_reports()
      self.assertEqual(1, _count_host_to_device_events(evts))

  @test_util.run_v2_only
  def testFitWithPiecewiseConstantDecayLearningRateSchedule(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      # Clear old reports
      ipu.ops.summary_ops.get_ipu_reports()

      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1], w=0.2)
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      lrs = keras.optimizer_v2.learning_rate_schedule.PiecewiseConstantDecay(
          boundaries=[8, 16], values=[0.001, 0.0005, 0.0001])
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=lrs)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      m.fit(test_dataset(length=72), epochs=1)

      # Ensure that we are not downloading the weights each time even though
      # the 'learning rate' hyper is being updated
      evts = ipu.ops.summary_ops.get_ipu_reports()
      self.assertEqual(1, _count_host_to_device_events(evts))

  @test_util.run_v2_only
  def testTrainPipelineWithLstm(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32),
                                       dtype=dtypes.int32,
                                       batch_size=1)

      with ipu.keras.PipelineStage(0):
        x = ipu.keras.layers.Embedding(8000, 128)(input_layer)

      with ipu.keras.PipelineStage(1):
        x = ipu.keras.layers.PopnnLSTM(128, dropout=0.2)(x)
        x = keras.layers.Dense(1, activation='sigmoid')(x)

      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      history = m.fit(test_language_dataset(length=72), epochs=3, verbose=0)

      losses = history.history['loss']
      self.assertTrue(losses[0] > losses[-1])

  @test_util.run_v2_only
  def testFitWithMetrics(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1], w=0.2)
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.0001)
      m.compile(opt, loss='mse', metrics=['accuracy'])

      # Fit the weights to the dataset
      history = m.fit(test_dataset(), steps_per_epoch=2, epochs=2)

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss', 'accuracy'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(type(history.history['accuracy']), list)
      self.assertEqual(len(history.history['loss']), 2)
      self.assertEqual(len(history.history['accuracy']), 2)
      self.assertEqual(type(history.history['loss'][0]), np.float64)
      self.assertEqual(type(history.history['loss'][1]), np.float64)
      self.assertEqual(type(history.history['accuracy'][0]), np.float32)
      self.assertEqual(type(history.history['accuracy'][1]), np.float32)

  @test_util.run_v2_only
  def testEval_CpuMatch(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1], w=0.2)
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=8)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      m.compile("sgd", loss='mse')

      # Fit the weights to the dataset
      result = m.evaluate(test_dataset(length=96))

      # The result is an aggregate of the loss
      self.assertEqual(len(result), 1)
      self.assertEqual(type(result), list)

    loss_cpu = keras.losses.mean_squared_error
    cpu_loss = run_model_on_cpu(
        self, simple_sequential_pipeline([32, 2], [0, 1], w=0.2), [], 12, 8,
        loss_cpu, None)
    cpu_loss = list(map(lambda x: x.numpy(), cpu_loss))
    cpu_loss = aggregate_cpu_out(training_utils.MetricsAggregator, cpu_loss)

    self.assertAllClose(result, cpu_loss, rtol=1e-5)

  @test_util.run_v2_only
  def testPredict_CpuMatch(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1], w=0.2)
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=8)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      # Generate predictions
      ipu_output = m.predict(test_inference_dataset(length=96))

      # The result is the Numpy array of concatenated output tensors
      self.assertEqual(type(ipu_output), np.ndarray)
      self.assertEqual(ipu_output.shape, (96, 2))

    cpu_out = run_model_on_cpu(
        self, simple_sequential_pipeline([32, 2], [0, 1], w=0.2), [], 12, 8,
        None, None)
    cpu_out = list(map(lambda x: x.numpy(), cpu_out))
    cpu_out = aggregate_cpu_out(training_utils.OutputsAggregator, cpu_out)

    self.assertAllClose(ipu_output, cpu_out)

  @test_util.run_v2_only
  def testFitWithNumpyData(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1], w=0.2)
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=8)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Input data
      input_x = np.full([72, 32], 1.0, dtype=np.single)
      input_y = np.full([72, 2], 0.2, dtype=np.single)

      # Fit the weights to the dataset
      history = m.fit(input_x, input_y, batch_size=1)

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 1)
      self.assertEqual(type(history.history['loss'][0]), np.float64)

  @test_util.run_v2_only
  def testEvalWithNumpyData(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1], w=0.2)
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=8)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Input data
      input_x = np.full([72, 32], 1.0, dtype=np.single)
      input_y = np.full([72, 2], 0.2, dtype=np.single)

      # Fit the weights to the dataset
      result = m.evaluate(input_x, input_y, batch_size=1)

      self.assertEqual(len(result), 1)
      self.assertEqual(type(result), list)

  @test_util.run_v2_only
  def testPredictWithNumpyData(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_pipeline(input_layer, [32, 2], [0, 1], w=0.2)
      m = ipu.keras.PipelineModel(inputs=input_layer,
                                  outputs=x,
                                  gradient_accumulation_count=8)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Input data
      input_x = np.full([96, 32], 1.0, dtype=np.single)

      # Generate predictions
      result = m.predict(input_x, batch_size=1)

      # The result is the Numpy array of concatenated output tensors
      self.assertEqual(type(result), np.ndarray)
      self.assertEqual(result.shape, (96, 2))

  @test_util.run_v2_only
  def testModelToDot(self):
    # This test is conditional on both `pydot` and `graphviz` being installed
    if vis_utils.check_pydot():
      strategy = ipu.ipu_strategy.IPUStrategy()
      with strategy.scope():
        # Initialize pipeline model
        input_layer = keras.layers.Input(shape=(32))
        x = simple_pipeline(input_layer, [32, 2], [0, 1], w=0.2)
        m = ipu.keras.PipelineModel(inputs=input_layer,
                                    outputs=x,
                                    gradient_accumulation_count=8)

        # Figure out expected layer labels and shapes
        _maybe_unpack = lambda x: x[0] if isinstance(x, list) else x
        _shape_to_str = lambda s: str(_maybe_unpack(s)).replace("None", "?")
        expected_nodes = {}
        for l in m.layers:
          expected_nodes[l.__class__.__name__] = [
              _shape_to_str(l.input_shape),
              _shape_to_str(l.output_shape)
          ]

        # Create the dot graph and extract node labels, shapes with regex
        dot_graph = vis_utils.model_to_dot(m, show_shapes=True)
        dot_nodes = {}
        for node in dot_graph.get_nodes():
          label = node.get_label()
          if label:
            layer_name = label.split('\n')[0].split(' ')[1]
            dot_nodes[layer_name] = re.findall(r"(\(.*?\))", label)

        # The model layers and dot nodes should be the same
        self.assertDictEqual(expected_nodes, dot_nodes)

  @test_util.run_v2_only
  def testPredictReplaceableLayers(self):
    def f():
      C = keras.initializers.Constant(0.1)
      input_layer = keras.layers.Input(10)

      with ipu.keras.PipelineStage(0):
        # Test Embedding.
        x = keras.layers.Embedding(10, 2,
                                   embeddings_initializer=C)(input_layer)
        x = keras.layers.Dense(2, kernel_initializer=C)(x)

        # Test Dropout.
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(20, kernel_initializer=C)(x)

      with ipu.keras.PipelineStage(1):
        # Test LSTM.
        x = keras.layers.LSTM(5, kernel_initializer=C)(x)

        # Test Layer Norm.
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dense(20, kernel_initializer=C)(x)
        x = keras.layers.Reshape((10, 2))(x)

        # Test GRU.
        x = keras.layers.GRU(5, kernel_initializer=C)(x)

      return input_layer, x

    # Create some test data.
    data = np.ones((96, 10), dtype=np.int32)

    # Compute IPU model output, uses layer replacement.
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      # Compute output with substitution.
      m = ipu.keras.PipelineModel(*f(),
                                  gradient_accumulation_count=8,
                                  layer_replacement=True)
      ipu_out = m.predict(data, batch_size=4)

      # Compute output without substitution.
      m_no_sub = ipu.keras.PipelineModel(*f(), gradient_accumulation_count=8)
      no_sub_out = m_no_sub.predict(data, batch_size=4)

    self.assertAllClose(ipu_out, no_sub_out)


if __name__ == '__main__':
  test.main()
