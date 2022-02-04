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
from absl.testing import parameterized

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.keras.optimizer_v2 import gradient_descent


def get_mnist_dataset(batch_size):
  mnist = keras.datasets.mnist

  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0

  # Add a channels dimension.
  x_train = x_train[..., np.newaxis]
  x_test = x_test[..., np.newaxis]

  x_train = x_train.astype('float32')
  y_train = y_train.astype('float32')
  x_test = x_test.astype('float32')
  y_test = y_test.astype('float32')

  train_ds = dataset_ops.DatasetV2.from_tensor_slices(
      (x_train, y_train)).batch(batch_size, drop_remainder=True).repeat()

  eval_ds = dataset_ops.DatasetV2.from_tensor_slices(
      (x_test, y_test)).batch(batch_size, drop_remainder=True).repeat()

  predict_ds = dataset_ops.DatasetV2.from_tensor_slices(x_test).batch(
      batch_size, drop_remainder=True).repeat()

  return train_ds, eval_ds, predict_ds


def simple_sequential_model(pipeline):
  m = keras.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(10,
                         activation='softmax',
                         kernel_initializer=keras.initializers.Constant(0.5),
                         bias_initializer='zeros')
  ])
  if pipeline:
    m.set_pipeline_stage_assignment([0, 1])
  return m


def simple_functional_model(pipeline):
  d = keras.layers.Input((28, 28))
  x = keras.layers.Flatten()(d)
  x = keras.layers.Dense(10,
                         activation='softmax',
                         kernel_initializer=keras.initializers.Constant(0.5),
                         bias_initializer='zeros')(x)
  m = keras.Model(d, x)
  if pipeline:
    assignments = m.get_pipeline_stage_assignment()
    assignments[0].pipeline_stage = 0
    assignments[1].pipeline_stage = 1
    m.set_pipeline_stage_assignment(assignments)
  return m


class CountingCallback(keras.callbacks.Callback):
  def __init__(self):
    super().__init__()
    self._train_batch_begin_count = 0
    self._train_batch_end_count = 0
    self._test_batch_begin_count = 0
    self._test_batch_end_count = 0
    self._predict_batch_begin_count = 0
    self._predict_batch_end_count = 0

  def on_train_batch_begin(self, *args, **kwargs):  # pylint: disable=arguments-differ,unused-argument
    self._train_batch_begin_count += 1

  def on_train_batch_end(self, *args, **kwargs):  # pylint: disable=arguments-differ,unused-argument
    self._train_batch_end_count += 1

  def on_test_batch_begin(self, *args, **kwargs):  # pylint: disable=arguments-differ,unused-argument
    self._test_batch_begin_count += 1

  def on_test_batch_end(self, *args, **kwargs):  # pylint: disable=arguments-differ,unused-argument
    self._test_batch_end_count += 1

  def on_predict_batch_begin(self, *args, **kwargs):  # pylint: disable=arguments-differ,unused-argument
    self._predict_batch_begin_count += 1

  def on_predict_batch_end(self, *args, **kwargs):  # pylint: disable=arguments-differ,unused-argument
    self._predict_batch_end_count += 1


class KerasAsynchronousCallbacksTest(test.TestCase, parameterized.TestCase):
  TESTCASES = [{
      "testcase_name": "sequential",
      "model_fn": simple_sequential_model,
      "replication_factor": 1,
      "pipelined": False,
  }, {
      "testcase_name": "functional",
      "model_fn": simple_functional_model,
      "replication_factor": 1,
      "pipelined": False,
  }, {
      "testcase_name": "sequential_replicated",
      "model_fn": simple_sequential_model,
      "replication_factor": 2,
      "pipelined": False,
  }, {
      "testcase_name": "functional_replicated",
      "model_fn": simple_functional_model,
      "replication_factor": 2,
      "pipelined": False,
  }, {
      "testcase_name": "sequential_pipelined",
      "model_fn": simple_sequential_model,
      "replication_factor": 2,
      "pipelined": True,
  }, {
      "testcase_name": "functional_pipelined",
      "model_fn": simple_functional_model,
      "replication_factor": 2,
      "pipelined": True,
  }]

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_v2_only
  def testCounting(self, model_fn, replication_factor, pipelined):
    ipus_in_model = 2 if pipelined else 1
    num_ipus = replication_factor * ipus_in_model
    tu.skip_if_not_enough_ipus(self, num_ipus)

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = num_ipus
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    batch_size = 12
    gradient_accumulation_steps = 16
    gradient_accumulation_steps_per_replica = (gradient_accumulation_steps //
                                               replication_factor)
    steps_per_epoch = 64
    epochs = 2

    train_ds, eval_ds, predict_ds = get_mnist_dataset(batch_size)
    cpu_train_ds, _, _ = get_mnist_dataset(batch_size *
                                           gradient_accumulation_steps)

    lr = 0.01
    optimizer = gradient_descent.SGD(learning_rate=lr)

    # Run on CPU - simulate gradient accumulation by just using a bigger batch
    # size but less steps per epoch.
    m = model_fn(False)
    m.compile(optimizer=optimizer,
              loss=keras.losses.SparseCategoricalCrossentropy())
    m.fit(cpu_train_ds,
          steps_per_epoch=steps_per_epoch // gradient_accumulation_steps,
          epochs=epochs)
    cpu_weights = m.weights

    cpu_eval = m.evaluate(eval_ds, steps=steps_per_epoch)
    cpu_predict = m.predict(predict_ds, steps=steps_per_epoch)

    lr /= replication_factor

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      optimizer = gradient_descent.SGD(learning_rate=lr)
      m = model_fn(pipelined)

      steps_per_execution = gradient_accumulation_steps_per_replica
      m.compile(optimizer=optimizer,
                loss=keras.losses.SparseCategoricalCrossentropy(),
                steps_per_execution=steps_per_execution)

      m.set_asynchronous_callbacks(True)
      if pipelined:
        m.set_pipelining_options(gradient_accumulation_steps_per_replica=
                                 gradient_accumulation_steps_per_replica,
                                 experimental_normalize_gradients=True)
      else:
        m.set_gradient_accumulation_options(
            gradient_accumulation_steps_per_replica=
            gradient_accumulation_steps_per_replica,
            experimental_normalize_gradients=True)
      cb = CountingCallback()
      m.fit(train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=2,
            callbacks=[cb])
      self.assertEqual(cb._train_batch_begin_count, steps_per_epoch * epochs)  # pylint: disable=protected-access
      self.assertEqual(cb._train_batch_end_count, steps_per_epoch * epochs)  # pylint: disable=protected-access
      ipu_weights = m.weights

      ipu_eval = m.evaluate(eval_ds, steps=steps_per_epoch, callbacks=[cb])
      self.assertEqual(cb._test_batch_begin_count, steps_per_epoch)  # pylint: disable=protected-access
      self.assertEqual(cb._test_batch_end_count, steps_per_epoch)  # pylint: disable=protected-access

      ipu_predict = m.predict(predict_ds,
                              steps=steps_per_epoch,
                              callbacks=[cb])
      self.assertEqual(cb._predict_batch_begin_count, steps_per_epoch)  # pylint: disable=protected-access
      self.assertEqual(cb._predict_batch_end_count, steps_per_epoch)  # pylint: disable=protected-access

      if pipelined:
        # Check that when accumulating outputs, outfeeds are not called per
        # step.
        m.set_pipelining_options(gradient_accumulation_steps_per_replica=
                                 gradient_accumulation_steps_per_replica,
                                 accumulate_outfeed=True,
                                 experimental_normalize_gradients=True)
        cb = CountingCallback()
        m.evaluate(eval_ds, steps=steps_per_epoch, callbacks=[cb])
        self.assertEqual(
            cb._test_batch_begin_count,  # pylint: disable=protected-access
            steps_per_epoch // (steps_per_execution * replication_factor))
        self.assertEqual(
            cb._test_batch_end_count,  # pylint: disable=protected-access
            steps_per_epoch // (steps_per_execution * replication_factor))

        # Predict should still get per step results.
        m.predict(predict_ds, steps=steps_per_epoch, callbacks=[cb])
        self.assertEqual(cb._predict_batch_begin_count, steps_per_epoch)  # pylint: disable=protected-access
        self.assertEqual(cb._predict_batch_end_count, steps_per_epoch)  # pylint: disable=protected-access

    atol = 1e-3 if replication_factor > 1 else 1e-5
    rtol = 1e-3 if replication_factor > 1 else 1e-5

    self.assertAllClose(cpu_weights, ipu_weights)
    self.assertAllClose(cpu_eval, ipu_eval, atol=atol, rtol=rtol)
    self.assertAllClose(cpu_predict, ipu_predict, atol=atol, rtol=rtol)


if __name__ == '__main__':
  test.main()
