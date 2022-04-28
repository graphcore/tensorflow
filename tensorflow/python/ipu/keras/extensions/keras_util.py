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
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.util import deprecation


@deprecation.deprecated(None, "Use `tf.keras` instead of `tf.python.keras`.")
def merge_into_batch_dimension(tensors, local_replication_factor):  # pylint: disable=missing-return-type-doc
  """Merges steps (and replication) into batch dimension

  Args:
    tensors (tf.Tensor): The tensors that need to be merged.
    local_replication_factor (int): The replication factor
    used to retrieve these tensors.
  Returns:
    `tf.Tensor` Tensors reshaped into the batch dimension.
  """
  def merge_fn(x):
    # Merge the steps, batches (and replication) dimensions.
    flat_shape = [x.shape[0] * x.shape[1]] + x.shape[2:]

    if local_replication_factor > 1:
      flat_shape = [flat_shape[0] * flat_shape[1]] + flat_shape[2:]

    return array_ops.reshape(x, flat_shape)

  return nest.map_structure(merge_fn, tensors)
