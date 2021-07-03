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
"""IPUDataHandler tests."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.ipu.keras.extensions import data_adapter
from tensorflow.python.platform import test


class DataHandlerTest(keras_parameterized.TestCase):
  def test_finite_dataset_with_steps_per_epoch(self):
    data = dataset_ops.Dataset.from_tensor_slices([0, 1, 2, 3]).batch(
        1, drop_remainder=True)
    # User can choose to only partially consume `Dataset`.
    data_handler = data_adapter.IPUDataHandler(data,
                                               initial_epoch=0,
                                               epochs=2,
                                               steps_per_epoch=2)
    self.assertEqual(data_handler.inferred_steps, 2)
    self.assertFalse(data_handler._adapter.should_recreate_iterator())  # pylint: disable=protected-access
    returned_data = []
    for _, iterator in data_handler.enumerate_epochs():
      epoch_data = []
      for _ in data_handler.steps():
        epoch_data.append(next(iterator).numpy())
      returned_data.append(epoch_data)
    self.assertEqual(returned_data, [[0, 1], [2, 3]])

  def test_finite_dataset_without_steps_per_epoch(self):
    data = dataset_ops.Dataset.from_tensor_slices([0, 1, 2]).batch(
        1, drop_remainder=True)
    data_handler = data_adapter.IPUDataHandler(data, initial_epoch=0, epochs=2)
    self.assertEqual(data_handler.inferred_steps, 3)
    returned_data = []
    for _, iterator in data_handler.enumerate_epochs():
      epoch_data = []
      for _ in data_handler.steps():
        epoch_data.append(next(iterator).numpy())
      returned_data.append(epoch_data)
    self.assertEqual(returned_data, [[0, 1, 2], [0, 1, 2]])

  def test_finite_dataset_with_steps_per_epoch_exact_size(self):
    data = dataset_ops.Dataset.from_tensor_slices([0, 1, 2, 3]).batch(
        1, drop_remainder=True)
    # If user specifies exact size of `Dataset` as `steps_per_epoch`,
    # create a new iterator each epoch.
    data_handler = data_adapter.IPUDataHandler(data,
                                               initial_epoch=0,
                                               epochs=2,
                                               steps_per_epoch=4)
    self.assertTrue(data_handler._adapter.should_recreate_iterator())  # pylint: disable=protected-access
    returned_data = []
    for _, iterator in data_handler.enumerate_epochs():
      epoch_data = []
      for _ in data_handler.steps():
        epoch_data.append(next(iterator).numpy())
      returned_data.append(epoch_data)
    self.assertEqual(returned_data, [[0, 1, 2, 3], [0, 1, 2, 3]])

  def test_infinite_dataset_with_steps_per_epoch(self):
    data = dataset_ops.Dataset.from_tensor_slices([0, 1, 2]).batch(
        1, drop_remainder=True).repeat()
    data_handler = data_adapter.IPUDataHandler(data,
                                               initial_epoch=0,
                                               epochs=2,
                                               steps_per_epoch=3)
    returned_data = []
    for _, iterator in data_handler.enumerate_epochs():
      epoch_data = []
      for _ in data_handler.steps():
        epoch_data.append(next(iterator).numpy())
      returned_data.append(epoch_data)
    self.assertEqual(returned_data, [[0, 1, 2], [0, 1, 2]])

  def test_unknown_cardinality_dataset_with_steps_per_epoch(self):
    ds = dataset_ops.DatasetV2.from_tensor_slices([0, 1, 2, 3, 4, 5, 6])
    filtered_ds = ds.filter(lambda x: x < 4)
    self.assertEqual(
        cardinality.cardinality(filtered_ds).numpy(), cardinality.UNKNOWN)

    # User can choose to only partially consume `Dataset`.
    data_handler = data_adapter.IPUDataHandler(filtered_ds,
                                               initial_epoch=0,
                                               epochs=2,
                                               steps_per_epoch=2)
    self.assertFalse(data_handler._adapter.should_recreate_iterator())  # pylint: disable=protected-access
    returned_data = []
    for _, iterator in data_handler.enumerate_epochs():
      epoch_data = []
      for _ in data_handler.steps():
        epoch_data.append(next(iterator))
      returned_data.append(epoch_data)
    returned_data = self.evaluate(returned_data)
    self.assertEqual(returned_data, [[0, 1], [2, 3]])
    self.assertEqual(data_handler.inferred_steps, 2)

  def test_unknown_cardinality_dataset_without_steps_per_epoch(self):
    ds = dataset_ops.DatasetV2.from_tensor_slices([0, 1, 2, 3, 4, 5, 6])
    filtered_ds = ds.filter(lambda x: x < 4)
    self.assertEqual(
        cardinality.cardinality(filtered_ds).numpy(), cardinality.UNKNOWN)

    with self.assertRaisesRegex(ValueError, "Could not infer the size of"):
      data_handler = data_adapter.IPUDataHandler(filtered_ds,
                                                 initial_epoch=0,
                                                 epochs=2)
      del data_handler

  def test_insufficient_data(self):
    ds = dataset_ops.DatasetV2.from_tensor_slices([0, 1])
    ds = ds.filter(lambda *args, **kwargs: True)
    data_handler = data_adapter.IPUDataHandler(ds,
                                               initial_epoch=0,
                                               epochs=2,
                                               steps_per_epoch=3)
    returned_data = []
    for _, iterator in data_handler.enumerate_epochs():
      epoch_data = []
      for _ in data_handler.steps():
        with data_handler.catch_stop_iteration():
          epoch_data.append(next(iterator))
      returned_data.append(epoch_data)
    returned_data = self.evaluate(returned_data)
    self.assertTrue(data_handler._insufficient_data)  # pylint: disable=protected-access
    self.assertEqual(returned_data, [[0, 1]])

  def test_numpy(self):
    x = np.array([0, 1, 2])
    y = np.array([0, 2, 4])
    sw = np.array([0, 4, 8])
    data_handler = data_adapter.IPUDataHandler(x=x,
                                               y=y,
                                               sample_weight=sw,
                                               batch_size=1,
                                               epochs=2)
    returned_data = []
    for _, iterator in data_handler.enumerate_epochs():
      epoch_data = []
      for _ in data_handler.steps():
        epoch_data.append(next(iterator))
      returned_data.append(epoch_data)
    returned_data = self.evaluate(returned_data)
    self.assertEqual(returned_data,
                     [[(0, 0, 0), (1, 2, 4),
                       (2, 4, 8)], [(0, 0, 0), (1, 2, 4), (2, 4, 8)]])

  def test_numpy_partial_batch(self):
    x = np.array([0, 1, 2])
    y = np.array([0, 2, 4])
    sw = np.array([0, 4, 8])

    with self.assertRaisesRegex(
        ValueError, "The provided set of data has a partial batch"):
      data_handler = data_adapter.IPUDataHandler(x=x,
                                                 y=y,
                                                 sample_weight=sw,
                                                 batch_size=2,
                                                 epochs=2)
      del data_handler

  def test_generator(self):
    def generator():
      for _ in range(2):
        for step in range(3):
          yield (ops.convert_to_tensor_v2_with_dispatch([step]),)

    with self.assertRaisesRegex(
        ValueError, r"The provided set of data contains a shape \(None,\)"):
      data_handler = data_adapter.IPUDataHandler(generator(),
                                                 epochs=2,
                                                 steps_per_epoch=3)
      del data_handler

  def test_class_weight_user_errors(self):
    with self.assertRaisesRegex(ValueError, 'to be a dict with keys'):
      data_adapter.IPUDataHandler(
          x=[[0], [1], [2]],
          y=[[2], [1], [0]],
          batch_size=1,
          sample_weight=[[1.], [2.], [4.]],
          class_weight={
              0: 0.5,
              1: 1.,
              3: 1.5  # Skips class `2`.
          })

    with self.assertRaisesRegex(ValueError, 'with a single output'):
      data_adapter.IPUDataHandler(x=np.ones((10, 1)),
                                  y=[np.ones((10, 1)),
                                     np.zeros((10, 1))],
                                  batch_size=2,
                                  class_weight={
                                      0: 0.5,
                                      1: 1.,
                                      2: 1.5
                                  })

  @parameterized.named_parameters(('numpy', True), ('dataset', False))
  def test_single_x_input_no_tuple_wrapping(self, use_numpy):
    x = np.ones((10, 1))

    if use_numpy:
      batch_size = 2
    else:
      x = dataset_ops.Dataset.from_tensor_slices(x).batch(2,
                                                          drop_remainder=True)
      batch_size = None

    data_handler = data_adapter.IPUDataHandler(x, batch_size=batch_size)
    for _, iterator in data_handler.enumerate_epochs():
      for _ in data_handler.steps():
        # Check that single x input is not wrapped in a tuple.
        d = next(iterator)
        self.assertIsInstance(d, ops.Tensor)
        self.assertDTypeEqual(d, np.float32)

  def test_size(self):
    x = np.ones((10, 1))
    x = dataset_ops.Dataset.from_tensor_slices(x).batch(2, drop_remainder=True)
    with self.assertRaisesRegex(
        ValueError,
        r"Your input does not have enough data. Make sure that your dataset or "
        r"generator can generate at least 12 batches \(currently it can only "
        r"generate 5 batches\)"):
      data_handler = data_adapter.IPUDataHandler(x,
                                                 epochs=2,
                                                 steps_per_epoch=6)
      del data_handler

  def test_steps_per_execution(self):
    x = np.ones((10, 1))
    with self.assertRaisesRegex(
        ValueError,
        r"The inferred number of steps per epoch \(6\) is not divisible by the "
        r"steps per execution \(4\)"):
      data_handler = data_adapter.IPUDataHandler(
          x,
          epochs=2,
          batch_size=2,
          steps_per_epoch=6,
          steps_per_execution=ops.convert_to_tensor(4))
      del data_handler


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
