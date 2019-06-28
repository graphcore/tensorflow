# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""
Popnn embedding operator
~~~~~~~~~~~~~~~~~~~~~~~~
"""

from functools import reduce
from operator import mul

from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging


def embedding_lookup(params,
                     ids,
                     name=None,
                     one_hot_threshold=2048,
                     min_encoding_size=1216):
  """Looks up `ids` in a list of embedding tensors.

    This is designed to be a drop-in replacement for the typical use cases with
    `tf.nn.embedding_lookup` for the IPU.

    Args:
        params: A single tensor representing the complete embedding tensor.
        ids: A `Tensor` with type `int32` containing the ids to be looked up in
             `params`.
        name: A name for the operation.
        one_hot_threshold: The threshold below which the embedding lookup will
                           become a one-hot with matmul.
        min_encoding_size: The minimum encoding size for the embedding. This is
                           used to decide whether to split the embedding tensor.
    Returns:
        A `Tensor` with the same type as the tensors in `params`.
    """
  name = name or "embedding_lookup"
  ids_shape = ids.shape
  M = reduce(mul, ids.shape, 1).value
  K = params.shape[0].value
  N = params.shape[1].value
  ids_flat = array_ops.reshape(ids, [M])

  # Handle the small case with a one-hot and matmul
  if K < one_hot_threshold:
    ids_one_hot = array_ops.one_hot(
        ids, K, name=name + "_one_hot", dtype=params.dtype)
    ids_one_hot = array_ops.reshape(ids_one_hot, [M, K])
    result = math_ops.matmul(ids_one_hot, params, name=name + "_lookup")
    return array_ops.reshape(result, list(ids.shape) + [N])

  # Handle a badly balanced case by splitting and applying two embedding lookups
  elif N < min_encoding_size:
    balance_factor = (min_encoding_size + N - 1) // N

    # Do we need to pad the input tensor?
    if K % balance_factor != 0:
      padding = balance_factor - (K % balance_factor)
      logging.warning(
          "Rebalancing of input tensor to embedding_lookup op named '" +
          str(name) + "' failed. Consider adding " + str(padding) +
          " rows to your embedding.")
      return embedding_ops.embedding_lookup(params, ids, name=name)

    # Reshape to distribute the tensor across more of the tiles
    params = array_ops.reshape(params,
                               [K // balance_factor, N * balance_factor])

    # This embedding lookup will get balance_factor more elements than desired
    rows = embedding_lookup(
        params,
        ids_flat // balance_factor,
        name=name + "_balanced",
        one_hot_threshold=one_hot_threshold,
        min_encoding_size=0)

    M1 = rows.shape[0].value
    N1 = rows.shape[1].value

    # Build new indices which extract the desired elements from the rows tensor
    ids1 = (math_ops.range(0, M1) * balance_factor) + (
        ids_flat % balance_factor)

    # Reshape the rows, so that a single embedding encoding is in each row
    rows = array_ops.reshape(rows, [M1 * balance_factor, N1 // balance_factor])

    # Extract the desired embedding elements
    result = embedding_lookup(
        rows,
        ids1,
        name=name + "_reduce",
        one_hot_threshold=one_hot_threshold,
        min_encoding_size=0)

    # Reshape back to the user shape
    return array_ops.reshape(result, list(ids_shape) + [N])

  # Fallback to the tf embedding lookup
  else:
    return embedding_ops.embedding_lookup(params, ids, name=name)
