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
"""Test for IPU Keras Pipelined model."""

import numpy as np
import pva
from tensorflow.python.ipu.config import IPUConfig

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ipu.tests import pipelining_test_util
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def test_dataset(length=None, batch_size=1, x_val=1.0, y_val=0.2):

  constant_d = constant_op.constant(x_val, shape=[32])
  constant_l = constant_op.constant(y_val, shape=[2])

  ds = dataset_ops.Dataset.from_tensors((constant_d, constant_l))
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


def test_inference_dataset(length=None, batch_size=1, x_val=1.0):

  constant_d = constant_op.constant(x_val, shape=[32])

  ds = dataset_ops.Dataset.from_tensors(constant_d)
  ds = ds.repeat(length)
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


def simple_model():
  return keras.Sequential([
      keras.layers.Dense(4),
      keras.layers.Dense(4),
      keras.layers.Dense(4),
      keras.layers.Dense(8),
  ])


def simple_pipeline():
  m = simple_model()
  m.set_pipeline_stage_assignment([0, 0, 0, 1])
  return m


def fixed_weight_model(dtype=None):
  return keras.Sequential([
      keras.layers.Dense(4,
                         name="layer0",
                         kernel_initializer=keras.initializers.Constant(0.1),
                         dtype=dtype),
      keras.layers.Dense(2,
                         name="layer1",
                         kernel_initializer=keras.initializers.Constant(0.1),
                         dtype=dtype),
  ])


def fixed_weight_pipeline(dtype=None):
  m = fixed_weight_model(dtype=dtype)
  m.set_pipeline_stage_assignment([0, 1])
  return m


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


class IPUSequentialPipelineTest(test.TestCase):
  @test_util.run_v2_only
  def testGradientAccumulationSteps(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()

    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    with strategy.scope():
      m = simple_pipeline()
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=7)
      m.compile('sgd', loss='mse', steps_per_execution=16)

      with self.assertRaisesRegex(
          RuntimeError,
          r"The pipelined model has been configured to use gradient "
          r"accumulation for training, however the current "
          r"`steps_per_execution` value \(set to 16\) is not divisible by "
          r"`gradient_accumulation_steps_per_replica \* number of replicas` "
          r"\(`gradient_accumulation_steps_per_replica` is set to 7 and there "
          r"are 1 replicas\)"):
        m.fit(test_dataset(length=64), epochs=4)

  @test_util.run_v2_only
  def testFitCpuMatch(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    gradient_accumulation_steps_per_replica = 8
    class_weight = {0: 0.0, 1: 0.1, 2: 0.9}

    # Run on CPU - simulate gradient accumulation by just using a bigger batch
    # size but less steps per epoch.
    m = fixed_weight_model()
    m.compile('sgd', loss='mse')
    m.fit(test_dataset(length=96,
                       batch_size=gradient_accumulation_steps_per_replica),
          epochs=2,
          class_weight=class_weight)
    cpu_weights = m.weights

    with strategy.scope():
      m = fixed_weight_pipeline()
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
      m = fixed_weight_pipeline()
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

      m = fixed_weight_pipeline()
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

      m = fixed_weight_pipeline()
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

      m = fixed_weight_pipeline()
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

      m = fixed_weight_pipeline()
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
      m = fixed_weight_pipeline()
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
      m = fixed_weight_pipeline()
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=8)
      m_acc = fixed_weight_pipeline()
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
  def testEval_CpuMatch(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = fixed_weight_pipeline()
      m.compile("sgd", loss='mse', steps_per_execution=32)
      # Evaluate the inputs using the fixed weight model
      result = m.evaluate(test_dataset(length=96))

    m = fixed_weight_model()
    m.compile("sgd", loss='mse', steps_per_execution=32)
    cpu_result = m.evaluate(test_dataset(length=96))

    self.assertAllClose(result, cpu_result)

  @test_util.run_v2_only
  def testPredict_CpuMatch(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = fixed_weight_pipeline()
      m.compile(steps_per_execution=32)
      # Evaluate the inputs using the fixed weight model
      result = m.evaluate(test_inference_dataset(length=96))

    m = fixed_weight_model()
    m.compile(steps_per_execution=32)
    cpu_result = m.evaluate(test_inference_dataset(length=96))

    self.assertAllClose(result, cpu_result)

  @test_util.run_v2_only
  def testGradientAccumulationDtype(self):
    def dtype_getter(var):
      self.assertEqual(var.dtype, np.float16)
      return np.float32

    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      policy.set_policy('float32')
      m = fixed_weight_pipeline(dtype=np.float16)
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=24,
                               gradient_accumulation_dtype=dtype_getter)

      outer = self

      class CastSGD(keras.optimizer_v2.gradient_descent.SGD):
        def apply_gradients(self, grads_and_vars, name=None):
          cast_grads_and_vars = []
          for (g, v) in grads_and_vars:
            outer.assertEqual(g.dtype, np.float32)
            cast_grads_and_vars.append((math_ops.cast(g, v.dtype), v))
          return super().apply_gradients(cast_grads_and_vars, name)

      opt = CastSGD()
      m.compile(opt, loss='mse', steps_per_execution=24)
      m.fit(test_dataset(x_val=np.float16(1.0),
                         y_val=np.float16(0.2),
                         length=48),
            epochs=1)


if __name__ == '__main__':
  test.main()
