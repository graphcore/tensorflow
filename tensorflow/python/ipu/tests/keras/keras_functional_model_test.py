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
"""Test for IPU Keras Model."""

import numpy as np

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


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


def test_dataset_two_input_output(length=None,
                                  batch_size=1,
                                  x_val=1.0,
                                  y_val=0.2):
  constant_d = constant_op.constant(x_val, shape=[32])
  constant_l = constant_op.constant(y_val, shape=[2])

  ds = dataset_ops.Dataset.from_tensors(({
      "input_a": constant_d,
      "input_b": constant_d
  }, {
      "target_a": constant_l,
      "target_b": constant_l
  }))
  ds = ds.repeat(length)
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


def test_inference_dataset_two_input_output(length=None,
                                            batch_size=1,
                                            x_val=1.0):
  constant_d = constant_op.constant(x_val, shape=[32])

  ds = dataset_ops.Dataset.from_tensors(({
      "input_a": constant_d,
      "input_b": constant_d
  }))
  ds = ds.repeat(length)
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


def simple_model(x, layer_sizes, w=None):
  assert layer_sizes

  init = 'glorot_uniform'
  if w:
    assert w > 0
    init = keras.initializers.Constant(w)

  y = keras.layers.Dense(layer_sizes[0],
                         activation=keras.activations.relu,
                         kernel_initializer=init)(x)

  for n in layer_sizes[1:]:
    y = keras.layers.Dense(n,
                           activation=keras.activations.relu,
                           kernel_initializer=init)(y)
  return y


def simple_cpu_model(layer_sizes, w=None):
  assert layer_sizes

  init = 'glorot_uniform'
  if w:
    assert w > 0
    init = keras.initializers.Constant(w)

  layers = []
  for n in layer_sizes:
    layers.append(
        keras.layers.Dense(n,
                           activation=keras.activations.relu,
                           kernel_initializer=init))
  return layers


def run_model_on_cpu(model, dataset, repeat_count, accumulation_count, loss,
                     optimizer):
  it = iter(dataset)
  results = []

  for _ in range(repeat_count):

    accumulations = None

    for _ in range(accumulation_count):
      x, t = next(it)

      with backprop.GradientTape() as tape:
        for l in model:
          x = l(x)

        if loss:
          l = loss(y_true=t, y_pred=x)
          results.append(l)
        else:
          results.append(x)

      if optimizer:
        all_vars = [v for layer in model for v in layer.trainable_variables]
        gradients = tape.gradient(l, all_vars)

        if not accumulations:
          accumulations = gradients
        else:
          accumulations = [a[0] + a[1] for a in zip(gradients, accumulations)]

    if optimizer:
      optimizer.apply_gradients(zip(accumulations, all_vars))

  return results


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


class IPUModelModelTest(test.TestCase):
  @test_util.run_v2_only
  def testModelCreation(self):
    # Simple single input, single output model.
    input_layer = keras.layers.Input(shape=(2))
    x = simple_model(input_layer, [2, 4])
    y = keras.layers.Activation(keras.activations.relu)(x)
    m = ipu.keras.Model(inputs=input_layer, outputs=y)

    # Verify dims.
    self.assertEqual(
        m._input_layers[0].get_output_at(0).get_shape().as_list(),  # pylint: disable=protected-access
        [None, 2])
    self.assertEqual(
        m._output_layers[0].get_output_at(0).get_shape().as_list(), [None, 4])  # pylint: disable=protected-access

  @test_util.run_v2_only
  def testModelCreationMultipleInput(self):
    # Simple two input, one output model.
    input_layer = keras.layers.Input(shape=(2))
    input_layer_two = keras.layers.Input(shape=(2))

    x = simple_model(input_layer, [2, 4])
    xx = simple_model(input_layer_two, [2, 4])

    x_con = keras.layers.concatenate([x, xx])
    y = keras.layers.Activation(keras.activations.relu)(x_con)

    m = ipu.keras.Model(inputs=[input_layer, input_layer_two], outputs=y)

    # Verify dims.
    self.assertEqual(len(m._input_layers), 2)  # pylint: disable=protected-access
    for d in m._input_layers:  # pylint: disable=protected-access
      self.assertEqual(d.get_output_at(0).get_shape().as_list(), [None, 2])
    self.assertEqual(
        m._output_layers[0].get_output_at(0).get_shape().as_list(), [None, 8])  # pylint: disable=protected-access

  @test_util.run_v2_only
  def testModelCreationMultipleOutput(self):
    # Simple one input, two output model.
    input_layer = keras.layers.Input(shape=(2))
    x = simple_model(input_layer, [2, 4])

    y = keras.layers.Activation(keras.activations.tanh)(x)
    yy = keras.layers.Activation(keras.activations.sigmoid)(x)

    m = ipu.keras.Model(inputs=input_layer, outputs=[y, yy])

    self.assertEqual(
        m._input_layers[0].get_output_at(0).get_shape().as_list(),  # pylint: disable=protected-access
        [None, 2])
    self.assertEqual(len(m._output_layers), 2)  # pylint: disable=protected-access
    for d in m._output_layers:  # pylint: disable=protected-access
      self.assertEqual(d.get_output_at(0).get_shape().as_list(), [None, 4])

  @test_util.run_v2_only
  def testCannotUseKerasV1Optimizers(self):
    input_layer = keras.layers.Input(shape=(2))
    x = simple_model(input_layer, [2, 4])
    m = ipu.keras.Model(inputs=input_layer, outputs=x)

    with self.assertRaisesRegex(
        ValueError,
        "Optimizer must be a native Tensorflow optimizers, or Keras V2"):
      opt = keras.optimizers.SGD(lr=0.001)
      m.compile(opt, 'mse')

  @test_util.run_v2_only
  def testMustCallCompileFit(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32])
      m = ipu.keras.Model(inputs=input_layer, outputs=x)

      with self.assertRaisesRegex(
          RuntimeError, "You must compile your model before training/testing"):
        m.fit(test_dataset())

  @test_util.run_v2_only
  def testMustCallCompileEvaluate(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32])
      m = ipu.keras.Model(inputs=input_layer, outputs=x)

      with self.assertRaisesRegex(
          RuntimeError, "You must compile your model before training/testing"):
        m.evaluate(test_dataset())

  @test_util.run_v2_only
  def testNeedTupleDatasetFit(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32])
      m = ipu.keras.Model(inputs=input_layer, outputs=x)
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError, r"requires a dataset containing a tuple of "):
        m.fit(test_inference_dataset())

  @test_util.run_v2_only
  def testNeedTupleDatasetEvaluate(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32])
      m = ipu.keras.Model(inputs=input_layer, outputs=x)
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError, r"requires a dataset containing a tuple of "):
        m.evaluate(test_inference_dataset())

  @test_util.run_v2_only
  def testNeedNonTupleDatasetPredict(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32])
      m = ipu.keras.Model(inputs=input_layer, outputs=x)

      with self.assertRaisesRegex(
          ValueError, r"requires a dataset containing a tuple of "):
        m.predict(test_dataset())

  @test_util.run_v2_only
  def testMismatchDatasetLengthToModelDepth(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 2])
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=24)

      m.compile('sgd', loss='mse')
      with self.assertRaisesRegex(
          ValueError, "Model requires the number of batches in the dataset"):
        m.fit(test_dataset(length=64), epochs=4)

  @test_util.run_v2_only
  def testUnlimitedDatasetHasNoStepsPerEpoch(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 2])
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=24)
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(
          ValueError, "When using an infinitely repeating dataset, you"):
        m.fit(test_dataset(), epochs=4)

  @test_util.run_v2_only
  def testStepsPerEpochTooLargeForDataset(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 2])
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=12)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      with self.assertRaisesRegex(
          ValueError,
          r"Steps per epoch times accumulation count \(14 x 12\) is greater "
          r"than"):
        m.fit(test_dataset(length=144), steps_per_epoch=14)

  @test_util.run_v2_only
  def testResultsOneEpochWithTfOptimizerNoAccumulation_CpuMatch(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 2], w=0.4)
      m = ipu.keras.Model(inputs=input_layer, outputs=x)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
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

    # Run the sequential CPU equivelant.
    m_cpu = simple_cpu_model([32, 32, 2], w=0.4)

    opt_cpu = gradient_descent.GradientDescentOptimizer(0.001)
    loss_cpu = keras.losses.mean_squared_error

    cpu_loss = run_model_on_cpu(m_cpu, test_dataset(length=96), 96, 1,
                                loss_cpu, opt_cpu)
    cpu_loss = list(map(lambda x: x.numpy(), cpu_loss))
    cpu_loss = aggregate_cpu_out(training_utils.MetricsAggregator, cpu_loss)
    cpu_loss = cpu_loss[0]

    # history['loss'] is one loss value per epoch (of which there is 1)
    ipu_loss = history.history['loss'][0]

    self.assertAllClose(ipu_loss, cpu_loss)

  @test_util.run_v2_only
  def testResultsOneEpochWithTfOptimizer_CpuMatch(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 2], w=0.4)
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=8)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
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

    # Run the sequential CPU equivelant.
    m_cpu = simple_cpu_model([32, 32, 2], w=0.4)

    opt_cpu = gradient_descent.GradientDescentOptimizer(0.001)
    loss_cpu = keras.losses.mean_squared_error
    cpu_loss = run_model_on_cpu(m_cpu, test_dataset(length=96), 12, 8,
                                loss_cpu, opt_cpu)
    cpu_loss = list(map(lambda x: x.numpy(), cpu_loss))
    cpu_loss = aggregate_cpu_out(training_utils.MetricsAggregator, cpu_loss)
    cpu_loss = cpu_loss[0]

    # history['loss'] is one loss value per epoch (of which there is 1)
    ipu_loss = history.history['loss'][0]

    self.assertAllClose(ipu_loss, cpu_loss)

  @test_util.run_v2_only
  def testFitWithTensorDataNoBatchSize(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 2], w=0.4)
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Input data
      input_x = constant_op.constant(1.0, shape=[72, 32])
      input_y = constant_op.constant(0.2, shape=[72, 2])

      # Fit the weights to the dataset
      with self.assertRaisesRegex(ValueError,
                                  "`batch_size` or `steps` is required for "):
        m.fit(input_x, input_y)

  @test_util.run_v2_only
  def testFitWithTensorData(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 2], w=0.4)
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

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
      self.assertEqual(type(history.history['loss'][0]), np.float64)

  @test_util.run_v2_only
  def testFitWithNumpyData(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 2], w=0.4)
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
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
      x = simple_model(input_layer, [32, 32, 2], w=0.4)
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
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
  def testPredictWithNumpyDataBs1(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 2], w=0.4)
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=12)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Input data
      input_x = np.full([96, 32], 1.0, dtype=np.single)

      # Fit the weights to the dataset
      result = m.predict(input_x, batch_size=1)

      self.assertEqual(type(result), tuple)
      self.assertEqual(len(result), 96)
      for i, r in enumerate(result):
        self.assertAllEqual(r, result[i - 1])

  @test_util.run_v2_only
  def testFitHistoryWithKerasOptimizer(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 2], w=0.4)
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
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
      x = simple_model(input_layer, [32, 32, 2], w=0.4)
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
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
      x = simple_model(input_layer, [32, 32, 2], w=0.4)
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # With a strategy saying 2 steps per run, and a step having a
      # accumulation_count=24 mini-batches, we should consume a 96 sample
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
      x = simple_model(input_layer, [32, 32, 2], w=0.4)
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
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
      x = simple_model(input_layer, [32, 32, 2])
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
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
      x = simple_model(input_layer, [32, 32, 2])
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
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
      x = simple_model(input_layer, [32, 32, 2])
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
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

      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 2])
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001,
                                                    decay=0.1)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      m.fit(test_dataset(length=72), epochs=2)

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
      x = simple_model(input_layer, [32, 32, 2])
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      lrs = keras.optimizer_v2.learning_rate_schedule.ExponentialDecay(
          0.001, 4, 0.1, staircase=True)
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=lrs)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      m.fit(test_dataset(length=72), epochs=2)

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
      x = simple_model(input_layer, [32, 32, 2])
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      lrs = keras.optimizer_v2.learning_rate_schedule.PiecewiseConstantDecay(
          boundaries=[8, 16], values=[0.001, 0.0005, 0.0001])
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=lrs)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      m.fit(test_dataset(length=72), epochs=2)

      # Ensure that we are not downloading the weights each time even though
      # the 'learning rate' hyper is being updated
      evts = ipu.ops.summary_ops.get_ipu_reports()
      self.assertEqual(1, _count_host_to_device_events(evts))

  @test_util.run_v2_only
  def testFitWithMetrics(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 2])
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=24)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
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
      self.assertEqual(type(history.history['loss'][0]), np.float64)
      self.assertEqual(len(history.history['accuracy']), 2)
      self.assertEqual(type(history.history['loss'][1]), np.float64)
      self.assertEqual(type(history.history['accuracy'][0]), np.float32)
      self.assertEqual(type(history.history['accuracy'][1]), np.float32)

  @test_util.run_v2_only
  def testEval_CpuMatch(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 2], w=0.4)
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=8)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      m.compile("sgd", loss='mse')

      # Fit the weights to the dataset
      result = m.evaluate(test_dataset(length=96))

      # The result is an aggregate of the loss
      self.assertEqual(len(result), 1)
      self.assertEqual(type(result), list)

    m_cpu = simple_cpu_model([32, 32, 2], w=0.4)
    loss_cpu = keras.losses.mean_squared_error
    cpu_loss = run_model_on_cpu(m_cpu, test_dataset(length=96), 12, 8,
                                loss_cpu, None)
    cpu_loss = list(map(lambda x: x.numpy(), cpu_loss))
    cpu_loss = aggregate_cpu_out(training_utils.MetricsAggregator, cpu_loss)

    self.assertAllClose(result, cpu_loss)

  @test_util.run_v2_only
  def testPredict_CpuMatch(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 2], w=0.4)
      m = ipu.keras.Model(inputs=input_layer, outputs=x, accumulation_count=8)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      # Fit the weights to the dataset
      ipu_out = m.predict(test_inference_dataset(length=96))

    m_cpu = simple_cpu_model([32, 32, 2], w=0.4)
    cpu_out = run_model_on_cpu(m_cpu, test_dataset(length=96), 12, 8, None,
                               None)
    cpu_out = list(map(lambda x: x.numpy(), cpu_out))
    cpu_out = aggregate_cpu_out(training_utils.OutputsAggregator, cpu_out)

    self.assertAllClose(ipu_out, cpu_out)

  @test_util.run_v2_only
  def testTrainMultipleInput(self):
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_a = keras.layers.Input(shape=(32))
      input_b = keras.layers.Input(shape=(32))

      block_a = simple_model(input_a, [32, 32], w=0.4)
      block_b = simple_model(input_b, [32, 32], w=0.4)

      concat_ab = keras.layers.concatenate([block_a, block_b])

      block_c = simple_model(concat_ab, [32, 2])
      block_d = simple_model(concat_ab, [32, 2])

      m = ipu.keras.Model(inputs=[input_a, input_b],
                          outputs=[block_c, block_d],
                          accumulation_count=8)

      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.auto_select_ipus(cfg, 1)
      ipu.utils.configure_ipu_system(cfg)

      m.compile("sgd", loss=['mse', 'mse'])

      ds = test_dataset_two_input_output(length=96, batch_size=4)
      m.fit(ds)


if __name__ == '__main__':
  test.main()
