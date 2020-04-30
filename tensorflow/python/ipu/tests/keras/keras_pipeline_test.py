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

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ipu.tests import pipelining_test_util
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


def next_feed_id():
  result = 'feed' + str(next_feed_id.feed_count)
  next_feed_id.feed_count += 1
  return result


next_feed_id.feed_count = 0


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


# Run a 2 stage model on the CPU, returning a single output value (loss
# or last stage output).
@def_function.function
def run_model_on_cpu(test_wrapper, model, input_values, repeat_count,
                     pipeline_depth, loss, optimizer):

  assert len(model) == 2

  def inputs_fn():
    return []

  def stage1(x, t):
    for l in model[0]:
      x = l(x)
    return x, t

  def stage2(x, t):
    for l in model[1]:
      x = l(x)

    if loss:
      return loss(y_true=t, y_pred=x)

    return x

  outputs = pipelining_test_util.PipelineTester.run_on_cpu(
      test_wrapper, [stage1, stage2], inputs_fn, input_values, repeat_count,
      pipeline_depth, test_dataset, optimizer)
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


def simple_pipeline():
  return [
      [
          keras.layers.Dense(4),
          keras.layers.Dense(4),
          keras.layers.Dense(4),
      ],
      [
          keras.layers.Dense(8),
      ],
  ]


def fixed_weight_pipeline():
  return [
      [
          keras.layers.Dense(
              4,
              name="layer0",
              kernel_initializer=keras.initializers.Constant(0.1)),
      ],
      [
          keras.layers.Dense(
              2,
              name="layer1",
              kernel_initializer=keras.initializers.Constant(0.1)),
      ],
  ]


def pipeline_with_lstm():
  return [[ipu.keras.layers.Embedding(8000, 128)],
          [
              ipu.keras.layers.PopnnLSTM(128, dropout=0.2),
              keras.layers.Dense(1, activation='sigmoid')
          ]]


def _count_host_to_device_events(evts):
  evt_types = ipu.utils.extract_all_types_from_event_trace(evts)
  evt_types = filter(lambda x: x == IpuTraceEvent.HOST_TO_DEVICE_TRANSFER,
                     evt_types)
  return len(list(evt_types))


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
  def testEmptyPipelineCreation(self):
    s = ipu.keras.PipelinedModel([], pipeline_depth=4)
    self.assertEqual(s.layers, [])

  @test_util.run_v2_only
  def testNoPipelineDepth(self):
    with self.assertRaisesRegex(ValueError,
                                "pipeline_depth parameter must be specified"):
      ipu.keras.PipelinedModel([])

  @test_util.run_v2_only
  def testPipelineCreation(self):
    m = ipu.keras.PipelinedModel(simple_pipeline(), pipeline_depth=4)
    self.assertEqual(len(m.layers), 4)

  @test_util.run_v2_only
  def testPipelineBadLayers(self):
    with self.assertRaisesRegex(ValueError,
                                " may only contain lists of stages,"):
      ipu.keras.PipelinedModel([
          keras.layers.Dense(8),
          keras.layers.Dense(8),
      ])

  @test_util.run_v2_only
  def testCannotCallEagerly(self):
    p = ipu.keras.PipelinedModel(simple_pipeline(), pipeline_depth=4)

    c = constant_op.constant(np.zeros([1, 12], dtype=np.float32))

    with self.assertRaisesRegex(
        ValueError, "PipelineModel can only be called through the"):
      p(c)

  @test_util.run_v2_only
  def testCannotUseKerasV1Optimizers(self):
    p = ipu.keras.PipelinedModel(simple_pipeline(), pipeline_depth=4)

    with self.assertRaisesRegex(
        ValueError,
        "Optimizer must be a native Tensorflow optimizers, or Keras V2"):
      opt = keras.optimizers.SGD(lr=0.001)
      p.compile(opt, 'mse')

  @test_util.run_v2_only
  def testMustCallCompileFit(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu.keras.PipelinedModel(simple_pipeline(), pipeline_depth=24)

      with self.assertRaisesRegex(
          RuntimeError, "You must compile your model before training/testing"):
        m.fit(test_dataset(length=64))

  @test_util.run_v2_only
  def testMustCallCompileEvaluate(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu.keras.PipelinedModel(simple_pipeline(), pipeline_depth=24)

      with self.assertRaisesRegex(
          RuntimeError, "You must compile your model before training/testing"):
        m.evaluate(test_dataset(length=64))

  @test_util.run_v2_only
  def testNeedTupleDatasetFit(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu.keras.PipelinedModel(simple_pipeline(), pipeline_depth=24)
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError,
          "PipelineModel.fit requires a dataset containing a tuple"):
        m.fit(test_inference_dataset(length=48))

  @test_util.run_v2_only
  def testNeedTupleDatasetEvaluate(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu.keras.PipelinedModel(simple_pipeline(), pipeline_depth=24)
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError,
          "PipelineModel.evaluate requires a dataset containing a tuple"):
        m.evaluate(test_inference_dataset(length=48))

  @test_util.run_v2_only
  def testNeedNonTupleDatasetPredict(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu.keras.PipelinedModel(simple_pipeline(), pipeline_depth=24)

      with self.assertRaisesRegex(
          ValueError,
          "PipelineModel.predict requires a dataset containing either"):
        m.predict(test_dataset(length=48))

  @test_util.run_v2_only
  def testMismatchDatasetLengthToPipelineDepth(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu.keras.PipelinedModel(simple_pipeline(), pipeline_depth=24)
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError,
          "PipelineModels require the number of batches in the dataset"):
        m.fit(test_dataset(length=64), epochs=4)

  @test_util.run_v2_only
  def testUnlimitedDatasetHasNoStepsPerEpoch(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu.keras.PipelinedModel(simple_pipeline(), pipeline_depth=24)
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError, "When using an infinitely repeating dataset, you"):
        m.fit(test_dataset(), epochs=4)

  @test_util.run_v2_only
  def testStepsPerEpochTooLargeForDataset(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu.keras.PipelinedModel(fixed_weight_pipeline(), pipeline_depth=12)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      with self.assertRaisesRegex(
          ValueError,
          r"Steps per epoch times accumulation count \(14 x 12\) is greater than"
      ):
        m.fit(test_dataset(length=144), steps_per_epoch=14)

  @test_util.run_v2_only
  def testResultsOneEpochWithTfOptimizer_CpuMatch(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu.keras.PipelinedModel(fixed_weight_pipeline(), pipeline_depth=8)

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
    cpu_loss = run_model_on_cpu(self, fixed_weight_pipeline(), [], 12, 8,
                                loss_cpu, opt_cpu)
    cpu_loss = list(map(lambda x: x.numpy(), cpu_loss))
    cpu_loss = aggregate_cpu_out(training_utils.MetricsAggregator, cpu_loss)
    cpu_loss = cpu_loss[0]

    # history['loss'] is one loss value per epoch (of which there is 1)
    ipu_loss = history.history['loss'][0]

    self.assertAllClose(ipu_loss, cpu_loss)

  @test_util.run_v2_only
  def testFitHistoryWithKerasOptimizer(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu.keras.PipelinedModel(fixed_weight_pipeline(), pipeline_depth=24)

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
      m = ipu.keras.PipelinedModel(fixed_weight_pipeline(), pipeline_depth=24)

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
      m = ipu.keras.PipelinedModel(fixed_weight_pipeline(), pipeline_depth=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # With a strategy saying 2 steps per run, and a step having a
      # pipeline_depth=24 mini-batches, we should consume a 96 sample
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
      m = ipu.keras.PipelinedModel(fixed_weight_pipeline(), pipeline_depth=24)

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

      m = ipu.keras.PipelinedModel(fixed_weight_pipeline(), pipeline_depth=8)

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
      m = ipu.keras.PipelinedModel(fixed_weight_pipeline(), pipeline_depth=24)

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
      m = ipu.keras.PipelinedModel(fixed_weight_pipeline(), pipeline_depth=24)

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
  def testFitWithLearningRateDecay(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      # Clear old reports
      ipu.ops.summary_ops.get_ipu_reports()

      m = ipu.keras.PipelinedModel(fixed_weight_pipeline(), pipeline_depth=24)

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

      m = ipu.keras.PipelinedModel(fixed_weight_pipeline(), pipeline_depth=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      lrs = keras.optimizer_v2.learning_rate_schedule.ExponentialDecay(
          0.001, 4, 0.1, staircase=True)
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=lrs)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      m.fit(test_dataset(length=72), epochs=6)

      # Ensure that we are not downloading the weights each time even though
      # the 'learning rate' hyper is being updated
      evts = ipu.ops.summary_ops.get_ipu_reports()
      self.assertEqual(1, _count_host_to_device_events(evts))

  @test_util.run_v2_only
  def testFitWithPiecewiseConstantDecayLearningRateSchedule(self):
    # Clear old reports
    ipu.ops.summary_ops.get_ipu_reports()

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      # Clear old reports
      ipu.ops.summary_ops.get_ipu_reports()

      m = ipu.keras.PipelinedModel(fixed_weight_pipeline(), pipeline_depth=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      lrs = keras.optimizer_v2.learning_rate_schedule.PiecewiseConstantDecay(
          boundaries=[8, 16], values=[0.001, 0.0005, 0.0001])
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=lrs)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      m.fit(test_dataset(length=72), epochs=6)

      # Ensure that we are not downloading the weights each time even though
      # the 'learning rate' hyper is being updated
      evts = ipu.ops.summary_ops.get_ipu_reports()
      self.assertEqual(1, _count_host_to_device_events(evts))

  @test_util.run_v2_only
  def testTrainPipelineWithLstm(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu.keras.PipelinedModel(pipeline_with_lstm(), pipeline_depth=24)

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
      m = ipu.keras.PipelinedModel(fixed_weight_pipeline(), pipeline_depth=24)

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
      m = ipu.keras.PipelinedModel(fixed_weight_pipeline(), pipeline_depth=8)

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
    cpu_loss = run_model_on_cpu(self, fixed_weight_pipeline(), [], 12, 8,
                                loss_cpu, None)
    cpu_loss = list(map(lambda x: x.numpy(), cpu_loss))
    cpu_loss = aggregate_cpu_out(training_utils.MetricsAggregator, cpu_loss)
    cpu_loss = cpu_loss[0]

    # result is the single aggregated loss value
    ipu_loss = result[0]

    self.assertEqual(ipu_loss, cpu_loss)

  @test_util.run_v2_only
  def testPredict_CpuMatch(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = ipu.keras.PipelinedModel(fixed_weight_pipeline(), pipeline_depth=8)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 2)
      ipu.utils.configure_ipu_system(cfg)

      # Fit the weights to the dataset
      result = m.predict(test_inference_dataset(length=96))

      # The result is the tuple of concatenated output tensors
      self.assertEqual(type(result), tuple)
      self.assertEqual(len(result), 1)
      self.assertEqual(type(result[0]), np.ndarray)
      self.assertEqual(result[0].shape, (96, 2))

    cpu_out = run_model_on_cpu(self, fixed_weight_pipeline(), [], 12, 8, None,
                               None)
    cpu_out = list(map(lambda x: x.numpy(), cpu_out))
    cpu_out = aggregate_cpu_out(training_utils.OutputsAggregator, cpu_out)

    # result is the predicted values
    ipu_output = result[0]

    self.assertAllEqual(ipu_output, cpu_out)


if __name__ == '__main__':
  test.main()
