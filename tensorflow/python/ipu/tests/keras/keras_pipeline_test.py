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

from functools import partial
from tensorflow.python.ipu.config import IPUConfig
import pva
import re

import numpy as np

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def simple_model(layer_sizes, layer_stages, w=None, pipeline=False):
  class DummyPipelineStage:
    def __init__(self, _stage):
      pass

    def __enter__(self):
      return self

    def __exit__(self, _exception_type, _value, _traceback):
      pass

  scope = ipu.keras.PipelineStage if pipeline else DummyPipelineStage
  assert layer_sizes
  assert len(layer_sizes) == len(layer_stages)

  init = 'glorot_uniform'
  if w:
    assert w > 0
    init = keras.initializers.Constant(w)

  x = keras.layers.Input(shape=(32))

  with scope(layer_stages[0]):
    y = keras.layers.Dense(layer_sizes[0],
                           activation=keras.activations.relu,
                           kernel_initializer=init)(x)

  for n, stage in zip(layer_sizes[1:], layer_stages[1:]):
    with scope(stage):
      y = keras.layers.Dense(n,
                             activation=keras.activations.relu,
                             kernel_initializer=init)(y)
  return keras.Model(x, y)


def simple_pipeline(layer_sizes, layer_stages, w=None):
  return simple_model(layer_sizes, layer_stages, w=w, pipeline=True)


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


class IPUPipelineTest(test.TestCase):
  @test_util.run_v2_only
  def testFitCpuMatch(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    gradient_accumulation_steps_per_replica = 8

    # Run on CPU - simulate gradient accumulation by just using a bigger batch
    # size but less steps per epoch.
    class_weight = {0: 0.0, 1: 0.1, 2: 0.9}
    m = simple_model([32, 2], [0, 1], w=0.2)
    m.compile('sgd', loss='mse')
    m.fit(test_dataset(length=96,
                       batch_size=gradient_accumulation_steps_per_replica),
          epochs=2,
          class_weight=class_weight)
    cpu_weights = m.weights

    with strategy.scope():
      m = simple_pipeline([32, 2], [0, 1], w=0.2)
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=8,
                               experimental_normalize_gradients=True)
      m.compile('sgd', loss='mse', steps_per_execution=16)
      m.fit(test_dataset(length=96), epochs=2, class_weight=class_weight)
      ipu_weights = m.weights
    self.assertAllClose(cpu_weights, ipu_weights)

  @test_util.run_v2_only
  def testFitHistoryStepsPerRun(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = simple_pipeline([32, 2], [0, 1], w=0.2)
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=8,
                               experimental_normalize_gradients=True)
      m.compile('sgd', loss='mse', steps_per_execution=16)
      m.fit(test_dataset(length=96), epochs=2)

      # Should be called per batch - there are 96 batches.
      cb = BatchCallbackCounter()

      # Fit the weights to the dataset
      m.fit(test_dataset(length=96), callbacks=[cb])

      # Should be called 96 / 16 times
      self.assertEqual(cb.count(), 6)

  @test_util.run_v2_only
  def testFitTwice(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      ds = test_dataset()

      m = simple_pipeline([32, 32, 2], [0, 1, 1])
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=8,
                               experimental_normalize_gradients=True)
      m.compile('sgd', loss='mse', steps_per_execution=16)
      history = m.fit(ds, steps_per_epoch=16)

      l = history.history['loss'][0]

      # # Record weights
      w_1 = [w.numpy() for w in m.weights]

      # Fit the weights to the dataset
      history = m.fit(ds, steps_per_epoch=16)

      # Loss should be different after second training.
      self.assertTrue(l > history.history['loss'][0])

      w_2 = [w.numpy() for w in m.weights]

      # Weights should be different too.
      for w1, w2 in zip(w_1, w_2):
        self.assertFalse(np.all(w1 == w2))

      # Should have compiled the graph once, and executed twice.
      self.assert_num_reports(report_helper, 1)
      report = pva.openReport(report_helper.find_report())
      self.assert_number_of_executions(report, 2)
      report_helper.clear_reports()

      # Fit the weights with a new dataset
      history = m.fit(test_dataset(), steps_per_epoch=16)

      # Loss should be different after second training.
      self.assertTrue(l > history.history['loss'][0])

      w_3 = [w.numpy() for w in m.weights]

      # Weights should be different too.
      for w2, w3 in zip(w_2, w_3):
        self.assertFalse(np.all(w2 == w3))

      # Don't need to compile the graph again.
      self.assert_num_reports(report_helper, 0)

  @test_util.run_v2_only
  def testFitMultipleOutputs(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
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

      m = keras.Model(inputs=input_layer, outputs=[y1, y2])
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=4)
      m.compile('sgd', loss='mse', steps_per_execution=4)

    # Fit the weights to the dataset
    dataset = test_dataset(length=144, batch_size=2)

    def d(x, y):
      return x, (y, y)

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
    cfg = IPUConfig()
    tu.enable_ipu_events(cfg)
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    report_json = tu.ReportJSON(self, eager_mode=True)

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      # Clear old reports
      report_json.reset()

      ds = test_dataset()

      m = simple_pipeline([32, 2], [0, 1], w=0.2)
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=8,
                               experimental_normalize_gradients=True)
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001,
                                                    decay=0.1)
      m.compile(opt, loss='mse', steps_per_execution=16)
      m.fit(ds, steps_per_epoch=32, epochs=6)

      # Ensure that we are only downloading the weights at the end of each
      # epoch.
      report_json.parse_log()
      report_json.assert_num_host_to_device_transfer_events(6)

  @test_util.run_v2_only
  def testFitWithExponentialDecayLearningRateSchedule(self):
    cfg = IPUConfig()
    tu.enable_ipu_events(cfg)
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    report_json = tu.ReportJSON(self, eager_mode=True)

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      # Clear old reports
      report_json.reset()

      ds = test_dataset()

      m = simple_pipeline([32, 2], [0, 1], w=0.2)
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=8,
                               experimental_normalize_gradients=True)
      lrs = keras.optimizer_v2.learning_rate_schedule.ExponentialDecay(
          0.001, 4, 0.1, staircase=True)
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=lrs)
      m.compile(opt, loss='mse', steps_per_execution=16)
      m.fit(ds, steps_per_epoch=32, epochs=6)

      # Ensure that we are only downloading the weights at the end of each
      # epoch.
      report_json.parse_log()
      report_json.assert_num_host_to_device_transfer_events(6)

  @test_util.run_v2_only
  def testFitWithPiecewiseConstantDecayLearningRateSchedule(self):
    cfg = IPUConfig()
    tu.enable_ipu_events(cfg)
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    report_json = tu.ReportJSON(self, eager_mode=True)

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      # Clear old reports
      report_json.reset()

      ds = test_dataset()

      m = simple_pipeline([32, 2], [0, 1], w=0.2)
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=8,
                               experimental_normalize_gradients=True)
      lrs = keras.optimizer_v2.learning_rate_schedule.PiecewiseConstantDecay(
          boundaries=[8, 16], values=[0.001, 0.0005, 0.0001])
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=lrs)
      m.compile(opt, loss='mse', steps_per_execution=16)
      m.fit(ds, steps_per_epoch=32, epochs=6)

      # Ensure that we are only downloading the weights at the end of each
      # epoch.
      report_json.parse_log()
      report_json.assert_num_host_to_device_transfer_events(6)

  @test_util.run_v2_only
  def testFitWithMetrics(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = simple_pipeline([32, 2], [0, 1], w=0.2)
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=8,
                               experimental_normalize_gradients=True)
      m.compile('sgd',
                loss='mse',
                metrics=['accuracy'],
                steps_per_execution=16)
      history = m.fit(test_dataset(), steps_per_epoch=16, epochs=2)

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss', 'accuracy'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(type(history.history['accuracy']), list)
      self.assertEqual(len(history.history['loss']), 2)
      self.assertEqual(len(history.history['accuracy']), 2)
      self.assertEqual(type(history.history['loss'][0]), float)
      self.assertEqual(type(history.history['loss'][1]), float)
      self.assertEqual(type(history.history['accuracy'][0]), float)
      self.assertEqual(type(history.history['accuracy'][1]), float)

  @test_util.run_v2_only
  def testFitAndEvaluateAccumulateOutfeed(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()

    # Accumulating the outfeeds shouldn't make a difference to the outputs.
    with strategy.scope():
      m = simple_pipeline([32, 2], [0, 1], w=0.2)
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=8)
      m_acc = simple_pipeline([32, 2], [0, 1], w=0.2)
      m_acc.set_pipelining_options(gradient_accumulation_steps_per_replica=8,
                                   accumulate_outfeed=True)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.0001)
      m.compile(opt, loss='mse', metrics=['accuracy'], steps_per_execution=16)
      m_acc.compile(opt,
                    loss='mse',
                    metrics=['accuracy'],
                    steps_per_execution=16)

      # Check that callbacks are called the right number of times.
      cb = BatchCallbackCounter()
      cb_acc = BatchCallbackCounter()

      # Call fit without accumulate_outfeed and check not accumulated
      history = m.fit(test_dataset(),
                      steps_per_epoch=16,
                      epochs=10,
                      callbacks=[cb])

      # Call fit with accumulate_outfeed and check accumulated
      history_acc = m_acc.fit(test_dataset(),
                              steps_per_epoch=16,
                              epochs=10,
                              callbacks=[cb_acc])
      self.assertAllClose(history.history, history_acc.history)
      self.assertEqual(cb.count(), cb_acc.count())

      cb = BatchCallbackCounter()
      cb_acc = BatchCallbackCounter()
      # Call evaluate without accumulate_outfeed and check not accumulated
      history = m.evaluate(test_dataset(length=96), callbacks=[cb])

      # Call evaluate with accumulate_outfeed and check accumulated
      history_acc = m_acc.evaluate(test_dataset(length=96), callbacks=[cb_acc])
      self.assertAllClose(history, history_acc)
      self.assertEqual(cb.count(), cb_acc.count())

  @test_util.run_v2_only
  def testFitAccumulateOutfeedSetDtype(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      init = keras.initializers.Constant(0.1)

      with ipu.keras.PipelineStage(0):
        y = keras.layers.Dense(32,
                               activation=keras.activations.relu,
                               kernel_initializer=init)(input_layer)

      with ipu.keras.PipelineStage(1):
        y = keras.layers.Dense(2,
                               activation=keras.activations.relu,
                               kernel_initializer=init)(y)
        # Add 100000 to make the loss too large for fp16.
        x = keras.layers.Lambda(
            lambda x: math_ops.cast(x + 100000, np.float16))(y)

      m = keras.Model(input_layer, x)
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.0001)
      m.compile(optimizer=opt,
                loss='mse',
                metrics=[
                    keras.metrics.MeanAbsoluteError(),
                    keras.metrics.RootMeanSquaredError()
                ],
                steps_per_execution=8)
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=8,
                               accumulate_outfeed=True)

      # With default types nothing overflows.
      history = m.fit(test_dataset(), steps_per_epoch=8, epochs=1)
      self.assertAllClose(history.history['loss'], [10002007040.0])
      self.assertAllClose(history.history['mean_absolute_error'],
                          [100010.046875])
      self.assertAllClose(history.history['root_mean_squared_error'],
                          [100010.046875])

      # Also accumulate the two metrics in fp16 and check they overflow too.
      def fp16metrics(var):
        if "PipelineStage:1" in var.name or "PipelineStage:2" in var.name:
          return np.float16
        return var.dtype

      m.set_pipelining_options(gradient_accumulation_steps_per_replica=8,
                               accumulate_outfeed=True,
                               accumulate_outfeed_dtype=fp16metrics)
      history = m.fit(test_dataset(), steps_per_epoch=8, epochs=1)

      self.assertAllEqual(history.history['loss'], [np.inf])
      self.assertAllEqual(history.history['mean_absolute_error'], [np.inf])
      self.assertAllClose(history.history['root_mean_squared_error'],
                          [99999.8046875])

  @test_util.run_v2_only
  def testEval_CpuMatch(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = simple_pipeline([32, 2], [0, 1], w=0.2)
      m.compile("sgd", loss='mse', steps_per_execution=32)
      # Evaluate the inputs using the fixed weight model
      result = m.evaluate(test_dataset(length=96))

    m = simple_model([32, 2], [0, 1], w=0.2)
    m.compile("sgd", loss='mse', steps_per_execution=32)
    cpu_result = m.evaluate(test_dataset(length=96))

    # A difference in precision can be found between running this
    # test using `ipu_model` and using actual hardware. For this
    # reason we set the relative tolerance to 1e-5.
    self.assertAllClose(result, cpu_result, rtol=1e-5)

  @test_util.run_v2_only
  def testPredict_CpuMatch(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = simple_pipeline([32, 2], [0, 1], w=0.2)
      m.compile(steps_per_execution=32)
      # Evaluate the inputs using the fixed weight model
      result = m.evaluate(test_inference_dataset(length=96))

    m = simple_model([32, 2], [0, 1], w=0.2)
    m.compile(steps_per_execution=32)
    cpu_result = m.evaluate(test_inference_dataset(length=96))

    self.assertAllClose(result, cpu_result)

  @test_util.run_v2_only
  def testUint8(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    dataset = dataset_ops.Dataset.from_tensor_slices(np.array(range(16)))
    dataset = dataset.map(lambda x: math_ops.cast(x, dtype=np.uint8)).batch(
        1, drop_remainder=True).batch(1, drop_remainder=True)

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      inputs = keras.layers.Input(shape=[1])

      with ipu.keras.PipelineStage(0):
        x = keras.layers.Lambda(lambda x: math_ops.cast(x, dtype=np.float16))(
            inputs)
        x = keras.layers.Dense(10, dtype=np.float16,
                               kernel_initializer='ones')(x)

      with ipu.keras.PipelineStage(1):
        x = keras.layers.Dense(1, dtype=np.float16,
                               kernel_initializer='ones')(x)

      m = keras.Model(inputs, x)
      m.compile(steps_per_execution=8)

      output = m.predict(dataset)
      self.assertEqual(output.shape, (16, 1))
      self.assertAllClose(output.flatten(), [n * 10 for n in range(16)])

  @test_util.run_v2_only
  def testFitWithReusedLayer(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      layer1 = keras.layers.Dense(32,
                                  activation=keras.activations.relu,
                                  kernel_initializer='glorot_uniform')
      layer2 = keras.layers.Dense(32,
                                  activation=keras.activations.relu,
                                  kernel_initializer='glorot_uniform')
      with ipu.keras.PipelineStage(0):
        x = layer1(input_layer)
      with ipu.keras.PipelineStage(1):
        x = layer2(x)
      with ipu.keras.PipelineStage(2):
        x = layer1(x)
      m = keras.Model(input_layer, x)
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=6,
                               device_mapping=[0, 1, 0])

      m.compile('sgd', loss='mse', metrics=['accuracy'], steps_per_execution=6)

      # Ensure fit runs through successfully.
      m.fit(test_language_dataset(), steps_per_epoch=6)

  @test_util.run_v2_only
  def testFitWithStagesDefinedForLayerAndNodes(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      with ipu.keras.PipelineStage(0):  # Define stage 0 for layer.
        layer = keras.layers.Dense(32,
                                   activation=keras.activations.relu,
                                   kernel_initializer='glorot_uniform')

      x = input_layer
      for stage in [0, 1, 2]:  # Define stages for nodes.
        with ipu.keras.PipelineStage(stage):
          x = layer(x)

      m = keras.Model(input_layer, x)
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=6,
                               device_mapping=[0, 0, 0])

      m.compile('sgd', loss='mse', metrics=['accuracy'], steps_per_execution=6)

      # Ensure fit runs through successfully.
      m.fit(test_language_dataset(), steps_per_epoch=6)

  @test_util.run_v2_only
  def testFitFailsWithSameLayerOnDifferentDevices(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      with ipu.keras.PipelineStage(0):  # Define stage 0 for layer.
        layer = keras.layers.Dense(32,
                                   activation=keras.activations.relu,
                                   kernel_initializer='glorot_uniform')

      x = input_layer
      for stage in [0, 1]:  # Define stages for nodes.
        with ipu.keras.PipelineStage(stage):
          x = layer(x)

      m = keras.Model(input_layer, x)
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=4,
                               device_mapping=[0, 1])
      m.compile('sgd', loss='mse', metrics=['accuracy'], steps_per_execution=4)

      with self.assertRaisesRegex(
          Exception,
          "an input can only be used by pipeline stages on the same IPU"):
        m.fit(test_language_dataset(), steps_per_epoch=4)

  @test_util.run_v2_only
  def testFitWithStagesDefinedOnlyForLayers(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      input_layer = keras.layers.Input(shape=(32))
      with ipu.keras.PipelineStage(0):  # Define stage 0 for layer.
        layer1 = keras.layers.Dense(32,
                                    activation=keras.activations.relu,
                                    kernel_initializer='glorot_uniform')
      with ipu.keras.PipelineStage(1):  # Define stage 0 for layer.
        layer2 = keras.layers.Dense(32,
                                    activation=keras.activations.relu,
                                    kernel_initializer='glorot_uniform')

      x = input_layer
      x = layer1(x)
      x = layer1(x)
      x = layer2(x)

      m = keras.Model(input_layer, x)
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=4,
                               device_mapping=[0, 0])

      m.compile('sgd', loss='mse', metrics=['accuracy'], steps_per_execution=4)

      # Ensure fit runs through successfully.
      m.fit(test_language_dataset(), steps_per_epoch=4)

  @test_util.run_v2_only
  def testPipelineTrainArgument(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    class TestLayer(keras.layers.Layer):
      def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training = None

      def call(self, x, training=None):  #pylint: disable=arguments-differ
        if training is None:
          training = K.learning_phase()
        self.training = training
        return x

    test_layer = TestLayer()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      with ipu.keras.PipelineStage(0):
        x = keras.layers.Dense(32)(input_layer)
      with ipu.keras.PipelineStage(1):
        x = test_layer(x)

      m = keras.Model(input_layer, x)
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=4,
                               device_mapping=[0, 0])

      m.compile('sgd', loss='mse', steps_per_execution=4)
      m.fit(test_language_dataset(), steps_per_epoch=4)

      self.assertEqual(test_layer.training, True)

  def testPredictWithPipelinedNonlinearGraph(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    # Tests that pipeline stages are built in the correct order by keras.
    # These nodes would have multiple potential orderings, if not for the
    # pipeline stage assignments.
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input1 = keras.layers.Input(32)
      input2 = keras.layers.Input(32)
      with ipu.keras.PipelineStage(0):
        x1 = keras.layers.Flatten()(input1)
      with ipu.keras.PipelineStage(1):
        x2 = keras.layers.Flatten()(input2)
      l = keras.layers.Dense(4)
      with ipu.keras.PipelineStage(2):
        x1 = l(x1)
      with ipu.keras.PipelineStage(3):
        x2 = l(x2)

      m = keras.Model((input1, input2), (x1, x2))
      m.set_pipelining_options(device_mapping=[0] * 4)
      inputs = [
          np.ones(shape=(4, 32), dtype=np.int32),
          np.ones(shape=(4, 32), dtype=np.int32)
      ]
      m.compile('sgd', loss='mse', steps_per_execution=4)
      # If the nodes were not built in the correct order, calling predict will
      # result in an exception.
      m.predict(inputs, batch_size=1)


if __name__ == '__main__':
  test.main()
