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
Popops embedding operators.
~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from functools import reduce
from operator import mul

from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import deprecation


@deprecation.deprecated_args(None, "stop passing this argument.",
                             "one_hot_threshold", "min_encoding_size")
def embedding_lookup(params,
                     ids,
                     name=None,
                     one_hot_threshold=0,
                     min_encoding_size=1216):
  """Looks up `ids` in a list of embedding tensors.

    This is designed to be a drop-in replacement for the typical use cases with
    `tf.nn.embedding_lookup` for the IPU.

    Args:
        params: A single tensor representing the complete embedding tensor.
        ids: A `Tensor` with type `int32` containing the slices to be extracted
             from `params`.
        name: A name for the operation.
        one_hot_threshold: The threshold below which the embedding lookup will
                           become a one-hot with matmul.
        min_encoding_size: The minimum encoding size for the embedding. This is
                           used to decide whether to split the embedding tensor.
    Returns:
        A `Tensor` with the same type as the tensors in `params`.
    """
  name = name or "embedding_lookup"
  ids_shape = ids.shape.as_list()
  params_shape = params.shape.as_list()

  # Flatten all the indices.
  num_ids = reduce(mul, ids_shape, 1)
  ids_flat = array_ops.reshape(ids, [num_ids])

  # Flatten params into a 2D shape.
  slice_dim_size = params_shape.pop(0)
  embedding_size = reduce(mul, params_shape, 1)
  params_2d = array_ops.reshape(params, [slice_dim_size, embedding_size])

  # Do the lookup.
  result = gen_popops_ops.ipu_multi_slice(params_2d, ids_flat, name=name)

  # Reshape into [ids[0], ... , ids[n - 1], params[1], ..., params[n - 1]]
  return array_ops.reshape(result, list(ids_shape) + list(params_shape))
