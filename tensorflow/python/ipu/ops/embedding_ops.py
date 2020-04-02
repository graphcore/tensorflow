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
Popops embedding operators
~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from functools import reduce
from operator import mul

from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import deprecation
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util

from tensorflow.compiler.plugin.poplar.ops import gen_pop_datastream_ops


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


class HostEmbeddingOptimizerSpec:
  """ Description of the Host Embedding optimizer.

      Despite the embedding living on the host, we want to compute the gradients
      on the device. Additionally, the communication channel between the device
      and host is opaque to TensorFlow. For these reasons we need to describe
      the optimiser parameters seperatenly.

      Currently only supports SGD.

      Args:
        learning_rate: The SGD learning rate.
  """
  def __init__(self, learning_rate):
    self._learning_rate = learning_rate

  def get_learning_rate(self):
    return self._learning_rate


class HostEmbedding:
  """ Host Embedding wrapper.

      HostEmbedding encapsulates the embedding tensor and the additional
      meta-data required to coordinate the host embedding and the device lookup.
      Through an instance of this class, an IPU can perform lookups on an
      embedding that resides on the host.

      It is assumed that the given embedding will be rank two where the
      outtermost dimension zero is the token dimension, and the innermost
      dimension is the encoding dimension.

      Args:
        name: The name which uniquely identifies the embedding.
        embedding_tensor: The tensor which holds the embedding.
        optimizer_spec: A description of how the embedding will be optimized.
                        When None, the embedding is assumed to not be trainable.
  """
  def __init__(self, name, embedding_tensor, optimizer_spec=None):
    if not tensor_util.is_tensor(embedding_tensor):
      raise ValueError(
          "HostEmbedding embedding_tensor is not a tensorflow tensor")

    if not isinstance(optimizer_spec,
                      (type(None), HostEmbeddingOptimizerSpec)):
      raise ValueError(
          "HostEmbedding optimizer_spec is not a HostEmbeddingOptimizerSpec" +
          " or None")

    self._name = name
    self._embedding_tensor = embedding_tensor
    self._lookup_count = 0
    self._update_count = 0
    self._optimizer_spec = optimizer_spec

  def __call__(self, iteration_count, replication_factor=1, training=True):
    """ Register the host embedding with the session.

        Args:
          iteration_count: The number of iterations in the user model.
          replication_factor: The replication count of the user graph.
          training: Whether this host embedding will be trained on this run.
                    This allows the user to specify that the embedding won't be
                    updated, despite the construction of gradient operations.
                    This is useful for validation, using the training graph.
        Returns:
          A TensorFlow op which will serve the embedding to the device.
    """
    if iteration_count <= 0:
      raise ValueError(
          "HostEmbedding call iteration count must be positive, but it is {}".
          format(iteration_count))
    return gen_pop_datastream_ops.ipu_host_embedding(
        self._embedding_tensor,
        self._name,
        lookup_count=iteration_count * self._lookup_count,
        update_count=(iteration_count * self._update_count if training else 0),
        replication_factor=replication_factor)

  def lookup(self, indices, count=1, clip_indices=True):
    """ Perform a host embedding lookup on an IPU.

        Args:
          indices: The indices to lookup.
          count: The number of times, per iteration, that this op will be
                 executed.
          clip_indices: Whether to enforce a the valid range on the lookup
                        indices with clipping. When False, out-of-range values
                        have undefined behaviour.
        Returns:
          A Tensor containing the elements requested by the user indices.
    """
    indices_shape = indices.shape.as_list()

    if clip_indices:
      indices = clip_ops.clip_by_value(indices, 0,
                                       self._embedding_tensor.shape[0] - 1)

    # Flatten all the indices.
    num_indices = reduce(mul, indices_shape, 1)
    indices_flat = array_ops.reshape(indices, [num_indices])

    self._lookup_count += count
    if self._optimizer_spec is not None:
      self._update_count += count
      with variable_scope.variable_scope(self._name,
                                         reuse=variable_scope.AUTO_REUSE):
        dummy = variable_scope.get_variable("__dummy",
                                            shape=[],
                                            dtype=self._embedding_tensor.dtype,
                                            trainable=True)
      result = gen_pop_datastream_ops.ipu_device_embedding_lookup_trainable(
          dummy,
          indices_flat,
          embedding_id=self._name,
          embedding_shape=self._embedding_tensor.shape,
          optimizer="SGD",
          learning_rate=self._optimizer_spec.get_learning_rate())
    else:
      result = gen_pop_datastream_ops.ipu_device_embedding_lookup(
          indices_flat,
          embedding_id=self._name,
          embedding_shape=self._embedding_tensor.shape,
          dtype=self._embedding_tensor.dtype)
    return array_ops.reshape(
        result,
        list(indices_shape) + [self._embedding_tensor.shape[1]])


def create_host_embedding(name,
                          shape,
                          dtype,
                          optimizer_spec=None,
                          initializer=None):
  with ops.device('cpu'):
    embedding_tensor = variable_scope.get_variable(name,
                                                   shape=shape,
                                                   dtype=dtype,
                                                   initializer=initializer)
  return HostEmbedding(name, embedding_tensor, optimizer_spec=optimizer_spec)
