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
"""
Embedding Keras layer
~~~~~~~~~~~~~~~~~~~~~
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import reduce
from operator import mul

from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export(v1=['keras.ipu.layers.Embedding'])
class Embedding(Layer):
  def __init__(self, name=None):
    """
    This is designed to be a replacement for the typical use cases of the
    Keras Embedding layer.

    Args:
        name: A name for the operation.
    """

    name = name or "embedding_lookup"

    super(Embedding, self).__init__(name=name)

  def saveable(self):
    raise NotImplementedError(
        "This cell does not yet support object-based saving. File a feature "
        "request if this limitation bothers you.")

  # pylint: disable=useless-super-delegation
  def build(self, input_shape):
    super(Embedding, self).build(input_shape)

  # pylint: disable=arguments-differ
  def call(self, params, ids):
    """
    Perform an embedding lookup.

    Args:
        params: An embedding tensor.
        ids: An integer tensor of indices into the params tensor.

    Returns:
        The entries of the embedding tensor corresponding to the ids tensor
        indices.
    """
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
    result = gen_popops_ops.ipu_multi_slice(params_2d,
                                            ids_flat,
                                            name=self.name)

    # Reshape into [ids[0], ... , ids[n - 1], params[1], ..., params[n - 1]]
    return array_ops.reshape(result, list(ids_shape) + list(params_shape))
