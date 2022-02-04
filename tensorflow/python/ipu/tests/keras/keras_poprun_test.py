# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

import popdist
import popdist.tensorflow

from tensorflow.python import ipu
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ipu.horovod import popdist_strategy
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu.ipu_outfeed_queue import IPUOutfeedQueue
from tensorflow.python.platform import test
from tensorflow.python.keras import layers, losses
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.callbacks import History
from tensorflow.python.types.core import Tensor


def test_dataset(length=None, batch_size=1, x_val=1.0, y_val=0.2):
  """Generates a flexible dataset that can be used for testing.

  Args:
      length (int, optional): Length of the dataset. Defaults to None.
      batch_size (int, optional): Batch size. Defaults to 1.
      x_val (float, optional): Default value for the inputs. Defaults to 1.0.
      y_val (float, optional): Default value for the labels. Defaults to 0.2.

  Returns:
      Dataset: The generated dataset.
  """
  constant_d = constant_op.constant(x_val, shape=[32])
  constant_l = constant_op.constant(y_val, shape=[1])

  ds = dataset_ops.Dataset.from_tensors((constant_d, constant_l))
  ds = ds.repeat(length)
  ds = ds.shard(num_shards=popdist.getNumInstances(),
                index=popdist.getInstanceIndex())
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


def simple_model():
  """Generates a simple reproducible model with just 1 dense layer.

  Returns:
      Model: The generated model.
  """
  class CustomModel(Model):  # pylint: disable=abstract-method
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

      self.loss_outfeed_queue = IPUOutfeedQueue()
      self.gradient_outfeed_queue = IPUOutfeedQueue()

    def train_step(self, data):
      data = data_adapter.expand_1d(data)
      x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

      with backprop.GradientTape() as tape:
        y_pred = self(x, training=True)
        loss = self.compiled_loss(y,
                                  y_pred,
                                  sample_weight,
                                  regularization_losses=self.losses)

      gradients = tape.gradient(loss, self.trainable_variables)

      self.loss_outfeed_queue.enqueue(loss)
      self.gradient_outfeed_queue.enqueue(gradients)

      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
      self.compiled_metrics.update_state(y, y_pred, sample_weight)

      return {m.name: m.result() for m in self.metrics}

    def get_latest_gradient(self):
      return self.gradient_outfeed_queue.dequeue()

    def get_latest_loss(self):
      return self.loss_outfeed_queue.dequeue()

  random_seed = 1234

  np.random.seed(random_seed)
  test_util.random_seed.set_seed(random_seed)

  inputs = Input(shape=(32,))
  outputs = layers.Dense(1)(inputs)

  return CustomModel(inputs, outputs)


def run_with_popdist(fn):
  """Run a provided function taking a `Model` as parameter using popdist.

  Args:
      fn (function): The function to run.

  Returns:
      mixed: Depends on the return value of `fn`.
  """
  config = ipu.config.IPUConfig()
  popdist.tensorflow.set_ipu_config(config, ipus_per_replica=1)
  config.configure_ipu_system()

  hvd.init()

  strategy = popdist_strategy.PopDistStrategy()

  with strategy.scope():
    model = simple_model()
    optimizer = gradient_descent.SGD(learning_rate=0.01)
    loss_fn = losses.MeanSquaredError()

    model.compile(optimizer=optimizer, loss=loss_fn, steps_per_execution=1)

    result = fn(model)

  if isinstance(result, Tensor):
    return model, (
        result,
        hvd.allgather(result),
    )

  return model, result


def run_with_horovod(fn):
  """Run a provided function taking a `Model` as parameter using horovod.

  Args:
      fn (function): The function to run.

  Returns:
      mixed: Depends on the return value of `fn`.
  """
  hvd.init()

  model = simple_model()
  optimizer = gradient_descent.SGD(learning_rate=0.01)
  loss_fn = losses.MeanSquaredError()

  model.compile(optimizer=optimizer, loss=loss_fn, steps_per_execution=1)

  result = fn(model)

  if isinstance(result, Tensor):
    return model, (
        result,
        hvd.allgather(result),
    )

  return model, result


# Different test callbacks.
def fit_on_full_dataset_with_fixed_size_one_epoch(model):
  return model.fit(test_dataset(8, batch_size=4), epochs=1, verbose=False)


def fit_on_full_dataset_with_fixed_size_two_epochs(model):
  return model.fit(test_dataset(8, batch_size=4), epochs=2, verbose=False)


def evaluate_on_full_dataset_with_fixed_size(model):
  return model.evaluate(test_dataset(100, batch_size=4), verbose=False)


def evaluate_on_full_dataset_with_fixed_size_with_fixed_steps(model):
  return model.evaluate(test_dataset(100, batch_size=4),
                        steps=2,
                        verbose=False)


def evaluate_on_full_dataset_without_fixed_size_with_fixed_steps(model):
  return model.evaluate(test_dataset(), steps=2, verbose=False)


def predict_on_full_dataset_with_fixed_size(model):
  return model.predict(test_dataset(100, batch_size=4), verbose=False)


def predict_on_full_dataset_with_fixed_size_with_fixed_steps(model):
  return model.predict(test_dataset(100, batch_size=4), steps=2, verbose=False)


def predict_on_full_dataset_without_fixed_size_with_fixed_steps(model):
  return model.predict(test_dataset(), steps=2, verbose=False)


TESTCASES = [
    (
        fit_on_full_dataset_with_fixed_size_one_epoch,
        True,
    ),
    (
        fit_on_full_dataset_with_fixed_size_two_epochs,
        True,
    ),
    (
        evaluate_on_full_dataset_with_fixed_size,
        False,
    ),
    (
        evaluate_on_full_dataset_with_fixed_size_with_fixed_steps,
        False,
    ),
    (
        evaluate_on_full_dataset_without_fixed_size_with_fixed_steps,
        False,
    ),
    (
        predict_on_full_dataset_with_fixed_size,
        False,
    ),
    (
        predict_on_full_dataset_with_fixed_size,
        False,
    ),
    (
        predict_on_full_dataset_with_fixed_size_with_fixed_steps,
        False,
    ),
    (
        predict_on_full_dataset_without_fixed_size_with_fixed_steps,
        False,
    ),
]


class KerasPoprunTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  def assert_gradients_are_equal(self, model_1, model_2):
    """Asserts that the last gradient from both models is identical.

    Args:
        model_1 (Model): First model to compare.
        model_2 (Model): Second model to compare.
    """
    # Ensure gradients before reduction are equal, but scaled by number of replicas.
    # This difference comes from the fact that the popdist version requires a strategy
    # which is based of the `CollectiveAllReduceStrategy`. The resulting weights
    # should *not* be different.
    num_replicas_in_sync = popdist.getNumInstances()

    self.assertAllClose(num_replicas_in_sync * model_1.get_latest_loss(),
                        model_2.get_latest_loss())

    popdist_latest_gradient = model_1.get_latest_gradient()
    horovod_latest_gradient = model_2.get_latest_gradient()

    for g_1, g_2 in zip(popdist_latest_gradient, horovod_latest_gradient):
      self.assertAllClose(num_replicas_in_sync * g_1, g_2)

  def assert_weight_updates_are_equal(self,
                                      model_1,
                                      model_2,
                                      did_weights_change=False):
    """Assert that the updated weights are identical between `model_1` and `model_2`.

    Args:
        model_1 (Model): First model to compare.
        model_2 (Model): Second model to compare.
        did_weights_change (bool, optional): Are we testing a training step.
        Defaults to False.
    """
    # Check that our trainable variables are the same after running `fn`.
    for v_1, v_2 in zip(model_1.trainable_variables,
                        model_2.trainable_variables):
      self.assertAllClose(v_1, v_2)

    # Create a vanilla model so we can compare whether the weights have been updated.
    vanilla_model = simple_model()
    vanilla_model.compile(loss=losses.MeanSquaredError())

    for v_1, v_2, v_vanilla in zip(model_1.trainable_variables,
                                   model_2.trainable_variables,
                                   vanilla_model.trainable_variables):
      if did_weights_change:
        self.assertNotAllClose(v_vanilla, v_1)
        self.assertNotAllClose(v_vanilla, v_2)
      else:
        self.assertAllClose(v_vanilla, v_1)
        self.assertAllClose(v_vanilla, v_2)

  def assert_result_is_equal(self, result_1, result_2):
    """Assert that the generated result of two separate models is identical.

    Args:
        result_1 (mixed): The first result.
        result_2 (mixed): The second result.
    """
    # Make sure that our returned result are of the same type.
    assert type(result_1) == type(result_2)  # pylint: disable=unidiomatic-typecheck

    # Ensure that the return value is equal.
    if isinstance(result_1, (
        Tensor,
        np.ndarray,
    )):
      self.assertAllClose(result_1, result_2)
    elif isinstance(result_1, History):
      for item_1, item_2 in zip(result_1.history.values(),
                                result_2.history.values()):
        self.assertAllClose(item_1, item_2)
    else:
      self.assertAllClose(result_1, result_2)

  @parameterized.parameters(*TESTCASES)
  def test_popdist_horovod_are_equal(self, callback, did_weights_change):
    """Tests whether the results of using keras from `keras_extensions_base`
    yields the same results as the upstream version after running a callback.

    Args:
        callback (function): The keras function to run.
        did_weights_change (bool): Assert that weights have been updated.
    """
    popdist_model, popdist_result = run_with_popdist(callback)
    horovod_model, horovod_result = run_with_horovod(callback)

    if did_weights_change:
      self.assert_gradients_are_equal(popdist_model, horovod_model)

    self.assert_weight_updates_are_equal(popdist_model, horovod_model,
                                         did_weights_change)
    self.assert_result_is_equal(popdist_result, horovod_result)


if __name__ == "__main__":
  test.main()
