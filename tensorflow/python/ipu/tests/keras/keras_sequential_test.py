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
"""Test for IPU Keras single IPU model."""

import pva
import numpy as np
from tensorflow.python.ipu.config import IPUConfig

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ipu.tests import pipelining_test_util
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


def test_dataset(length=None,
                 batch_size=1,
                 x_val=1.0,
                 y_val=0.2,
                 dtype=np.float32):

  constant_d = constant_op.constant(x_val, shape=[32], dtype=dtype)
  constant_l = constant_op.constant(y_val, shape=[2], dtype=dtype)

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


def test_inference_dataset(length=None,
                           batch_size=1,
                           x_val=1.0,
                           dtype=np.float32):

  constant_d = constant_op.constant(x_val, shape=[32], dtype=dtype)

  ds = dataset_ops.Dataset.from_tensors(constant_d)
  ds = ds.repeat(length)
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


def simple_model():
  return [
      keras.layers.Dense(4),
      keras.layers.Dense(8),
  ]


def fixed_weight_model():
  return [
      keras.layers.Dense(4,
                         name="layer0",
                         kernel_initializer=keras.initializers.Constant(0.1)),
      keras.layers.Dense(2,
                         name="layer1",
                         kernel_initializer=keras.initializers.Constant(0.1)),
  ]


class IPUModelTest(test.TestCase):
  @test_util.run_v2_only
  def testEmptyModelCreation(self):
    s = keras.Sequential([])
    self.assertEqual(s.layers, [])

  @test_util.run_v2_only
  def testModelCreation(self):
    m = keras.Sequential(simple_model())
    self.assertEqual(len(m.layers), 2)

  @test_util.run_v2_only
  def testModelBadLayers(self):
    with self.assertRaisesRegex(TypeError,
                                "The added layer must be an instance"):
      keras.Sequential([[
          keras.layers.Dense(8),
          keras.layers.Dense(8),
      ]])

  @test_util.run_v2_only
  def testMustCallCompileFit(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(simple_model())

      with self.assertRaisesRegex(
          RuntimeError, "You must compile your model before training/testing"):
        m.fit(test_dataset(length=64))

  @test_util.run_v2_only
  def testMustCallCompileEvaluate(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(simple_model())

      with self.assertRaisesRegex(
          RuntimeError, "You must compile your model before training/testing"):
        m.evaluate(test_dataset(length=64))

  @test_util.run_v2_only
  def testUnlimitedDatasetHasNoStepsPerEpoch(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(simple_model())
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(ValueError,
                                  "When providing an infinite dataset"):
        m.fit(test_dataset(), epochs=4)

  @test_util.run_v2_only
  def testBuildSequentialModel(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(simple_model())

      self.assertAllEqual(m.built, False)
      for l in m.layers:
        self.assertAllEqual(l.built, False)

      m.build(input_shape=(4, 32))

      self.assertAllEqual(m.built, True)
      for l in m.layers:
        self.assertAllEqual(l.built, True)

  @test_util.run_v2_only
  def testResultsOneEpochWithTfOptimizerNoAccumulation_CpuMatch(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = gradient_descent.GradientDescentOptimizer(0.001)

      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      history = m.fit(test_dataset(length=96))

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 1)
      self.assertEqual(type(history.history['loss'][0]), float)

    m_cpu = keras.Sequential(fixed_weight_model())
    opt_cpu = gradient_descent.GradientDescentOptimizer(0.001)
    m_cpu.compile(opt_cpu, loss='mse')
    cpu_history = m_cpu.fit(test_dataset(length=96))

    self.assertAllClose(history.history, cpu_history.history)

  @test_util.run_v2_only
  def testFitHistoryWithKerasOptimizer(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      history = m.fit(test_dataset(length=72))

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 1)
      self.assertEqual(type(history.history['loss'][0]), float)

  @test_util.run_v2_only
  def testFitHistoryTwoEpochs(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      history = m.fit(test_dataset(length=72), epochs=2)

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 2)
      self.assertEqual(type(history.history['loss'][0]), float)
      self.assertEqual(type(history.history['loss'][1]), float)

  @test_util.run_v2_only
  def testFitHistoryStepsPerEpochOneEpoch(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      history = m.fit(test_dataset(), steps_per_epoch=144)

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 1)
      self.assertEqual(type(history.history['loss'][0]), float)

  @test_util.run_v2_only
  def testFitTwice(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      ds = test_dataset()

      m = keras.Sequential(fixed_weight_model())

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      report_helper.clear_reports()

      # Fit the weights to the dataset
      history = m.fit(ds, steps_per_epoch=1)
      l = history.history['loss'][0]

      # # Record weights
      w_1 = [w.numpy() for w in m.weights]

      # Fit the weights to the dataset
      history = m.fit(ds, steps_per_epoch=1)

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
      history = m.fit(test_dataset(), steps_per_epoch=1)

      # Loss should be different after second training.
      self.assertTrue(l > history.history['loss'][0])

      w_3 = [w.numpy() for w in m.weights]

      # Weights should be different too.
      for w2, w3 in zip(w_2, w_3):
        self.assertFalse(np.all(w2 == w3))

      # Don't need to compile the graph again.
      self.assert_num_reports(report_helper, 0)

  @test_util.run_v2_only
  def testFitHistoryStepsPerEpochTwoEpochs(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      history = m.fit(test_dataset(), steps_per_epoch=144, epochs=2)

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 2)
      self.assertEqual(type(history.history['loss'][0]), float)
      self.assertEqual(type(history.history['loss'][1]), float)

  @test_util.run_v2_only
  def testFitWithLearningRateDecay(self):
    report_json = tu.ReportJSON(self, eager_mode=True)
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      # Clear old reports
      report_json.reset()

      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001,
                                                    decay=0.1)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      m.fit(test_dataset(length=72), epochs=4)

      # Ensure that we are only downloading the weights at the end of each
      # epoch.
      report_json.parse_log()
      report_json.assert_num_host_to_device_transfer_events(4)

  @test_util.run_v2_only
  def testFitWithExponentialDecayLearningRateSchedule(self):
    report_json = tu.ReportJSON(self, eager_mode=True)
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      # Clear old reports
      report_json.reset()

      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      lrs = keras.optimizer_v2.learning_rate_schedule.ExponentialDecay(
          0.001, 4, 0.1, staircase=True)
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=lrs)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      m.fit(test_dataset(length=72), epochs=4)

      # Ensure that we are only downloading the weights at the end of each
      # epoch.
      report_json.parse_log()
      report_json.assert_num_host_to_device_transfer_events(4)

  @test_util.run_v2_only
  def testFitWithPiecewiseConstantDecayLearningRateSchedule(self):
    report_json = tu.ReportJSON(self, eager_mode=True)
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      # Clear old reports
      report_json.reset()

      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      lrs = keras.optimizer_v2.learning_rate_schedule.PiecewiseConstantDecay(
          boundaries=[8, 16], values=[0.001, 0.0005, 0.0001])
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=lrs)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      m.fit(test_dataset(length=72), epochs=4)

      # Ensure that we are only downloading the weights at the end of each
      # epoch.
      report_json.parse_log()
      report_json.assert_num_host_to_device_transfer_events(4)

  @test_util.run_v2_only
  def testFitWithMetrics(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

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
      self.assertEqual(type(history.history['loss'][0]), float)
      self.assertEqual(type(history.history['loss'][1]), float)
      self.assertEqual(type(history.history['accuracy'][0]), float)
      self.assertEqual(type(history.history['accuracy'][1]), float)

  @test_util.run_v2_only
  def testEval_CpuMatch(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      m.compile("sgd", loss='mse')

      # Fit the weights to the dataset
      result = m.evaluate(test_dataset(length=96))

    m_cpu = keras.Sequential(fixed_weight_model())
    m_cpu.compile("sgd", loss='mse')
    cpu_result = m_cpu.evaluate(test_dataset(length=96))

    self.assertAllClose(result, cpu_result)

  @test_util.run_v2_only
  def testPredictBs1_CpuMatch(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      # Generate predictions
      ipu_out = m.predict(test_inference_dataset(length=96))

      # The result is the Numpy array of concatenated output tensors
      self.assertEqual(type(ipu_out), np.ndarray)
      self.assertEqual(ipu_out.shape, (96, 2))

    m_cpu = keras.Sequential(fixed_weight_model())
    cpu_out = m_cpu.predict(test_inference_dataset(length=96))

    self.assertAllClose(ipu_out, cpu_out)

  @test_util.run_v2_only
  def testPredictBs2_CpuMatch(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      # Generate predictions
      ipu_out = m.predict(test_inference_dataset(length=96, batch_size=2))

      # The result is the Numpy array of concatenated output tensors
      self.assertEqual(type(ipu_out), np.ndarray)
      self.assertEqual(ipu_out.shape, (96, 2))

    m_cpu = keras.Sequential(fixed_weight_model())
    cpu_out = m_cpu.predict(test_inference_dataset(length=96))

    self.assertAllClose(ipu_out, cpu_out)

  @test_util.run_v2_only
  def testFitWithTensorDataPartialBatch(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Input data
      input_x = constant_op.constant(1.0, shape=[72, 32])
      input_y = constant_op.constant(0.2, shape=[72, 2])

      # Fit the weights to the dataset
      with self.assertRaisesRegex(
          ValueError, "The provided set of data has a partial batch"):
        m.fit(input_x, input_y)

  @test_util.run_v2_only
  def testFitWithTensorData(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Input data
      input_x = constant_op.constant(1.0, shape=[72, 32])
      input_y = constant_op.constant(0.2, shape=[72, 2])

      # Fit the weights to the dataset
      history = m.fit(input_x, input_y, batch_size=1)

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 1)
      self.assertEqual(type(history.history['loss'][0]), float)

  @test_util.run_v2_only
  def testFitWithNumpyData(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

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
      self.assertEqual(type(history.history['loss'][0]), float)

  @test_util.run_v2_only
  def testEvalWithNumpyData(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Input data
      input_x = np.full([72, 32], 1.0, dtype=np.single)
      input_y = np.full([72, 2], 0.2, dtype=np.single)

      # Fit the weights to the dataset
      result = m.evaluate(input_x, input_y, batch_size=1)
      self.assertAllClose(result, 1.1664)

  @test_util.run_v2_only
  def testEvalWithNumpyDataBs2(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Input data
      input_x = np.full([72, 32], 1.0, dtype=np.single)
      input_y = np.full([72, 2], 0.2, dtype=np.single)

      # Fit the weights to the dataset
      result = m.evaluate(input_x, input_y, batch_size=2)
      self.assertAllClose(result, 1.1664)

  @test_util.run_v2_only
  def testPredictWithNumpyDataBs1(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Input data
      input_x = np.full([96, 32], 1.0, dtype=np.single)

      # Get predictions
      result = m.predict(input_x, batch_size=1)

      # The result is the Numpy array of concatenated output tensors
      self.assertEqual(type(result), np.ndarray)
      self.assertEqual(result.shape, (96, 2))

  @test_util.run_v2_only
  def testPredictWithNumpyDataBs2(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Input data
      input_x = np.full([96, 32], 1.0, dtype=np.single)

      # Generate predictions
      result = m.predict(input_x, batch_size=2)

      # The result is the Numpy array of concatenated output tensors
      self.assertEqual(type(result), np.ndarray)
      self.assertEqual(result.shape, (96, 2))

  @test_util.run_v2_only
  def testAutocast_V2DtypeBehaviourTrue(self):
    base_layer_utils.enable_v2_dtype_behavior()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Input data
      input_x = np.full([72, 32], 1.0, dtype=np.float64)
      input_y = np.full([72, 2], 0.2, dtype=np.float64)

      m.fit(input_x, input_y, batch_size=2)
      m.predict(input_x, batch_size=2)
      m.evaluate(input_x, input_y, batch_size=2)

      # No exceptions thrown

  @test_util.run_v2_only
  def testAutocast_V2DtypeBehaviourFalse(self):
    base_layer_utils.disable_v2_dtype_behavior()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Sequential(fixed_weight_model())

      cfg = IPUConfig()
      cfg._profiling.enable_ipu_events = True  # pylint: disable=protected-access
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      with self.assertRaisesWithPredicateMatch(errors.FailedPreconditionError,
                                               "Unsupported datatype double"):
        m.predict(test_inference_dataset(length=96, dtype=np.float64),
                  batch_size=2)

      with self.assertRaisesWithPredicateMatch(errors.FailedPreconditionError,
                                               "Unsupported datatype double"):
        m.evaluate(test_dataset(length=96, dtype=np.float64), batch_size=2)

      with self.assertRaisesWithPredicateMatch(errors.FailedPreconditionError,
                                               "Unsupported datatype double"):
        m.fit(test_dataset(length=96, dtype=np.float64), batch_size=2)


if __name__ == '__main__':
  test.main()
