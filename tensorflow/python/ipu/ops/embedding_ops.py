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
from tensorflow.python.ipu.ops import functional_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.eager import context

from tensorflow.compiler.plugin.poplar.ops import gen_pop_datastream_ops


def embedding_lookup(params, ids, name=None, serialization_factor=1):
  """Looks up `ids` in a list of embedding tensors.

    This is designed to be a drop-in replacement for the typical use cases with
    `tf.nn.embedding_lookup` for the IPU.

    Args:
        params: A single tensor representing the complete embedding tensor.
        ids: A `Tensor` with type `int32` containing the slices to be extracted
             from `params`.
        name: A name for the operation.
        serialization_factor: If greater than 1, the embedding lookup will be
             broken up into `serialization_factor` smaller lookups, serialized
             along the 0th dimension. This option should not be used unless
             `params` is used by another operation, such as matrix
             multiplication. If `params` has multiple users, then serialization
             can reduce the maximum memory at the cost of extra computation.
    Returns:
        A `Tensor` with the same type as the tensors in `params`.
    """
  serialization_factor = int(serialization_factor)
  if serialization_factor < 1:
    raise ValueError(
        'serialization_factor has to be at least 1, but was {}.'.format(
            serialization_factor))

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

  if (slice_dim_size % serialization_factor) != 0:
    raise ValueError(
        'The serialization_factor ({}) must divide the size of the 0th '
        'dimension of params ({}).'.format(serialization_factor,
                                           slice_dim_size))

  # Do the lookup.
  if serialization_factor == 1:
    result = gen_popops_ops.ipu_multi_slice(params_2d, ids_flat, name=name)
  else:
    # Get the scope name so that the function nested operations get the scope
    # name too.
    with ops.name_scope(name + "/Serialized") as scope_name:

      @custom_gradient.custom_gradient
      def serialized_embedding_lookup(table, indices):

        table_shape = table.shape.as_list()
        assert len(table_shape) == 2
        assert (table_shape[0] % serialization_factor) == 0
        split_size = table_shape[0] // serialization_factor

        @functional_ops.outlined_function(keep_input_layouts=False)
        def func(sliced_table, indices, min_idx):
          with ops.name_scope(scope_name):
            # Do a serialized embedding lookup by adjusting the indices.
            adjusted_indices = indices - min_idx

            # Mask out any indices which are not in range.
            mask_max = adjusted_indices < split_size
            mask_min = adjusted_indices > -1
            mask = math_ops.logical_and(mask_max, mask_min)

            indices_mask = math_ops.cast(mask, adjusted_indices.dtype)
            adjusted_indices = adjusted_indices * indices_mask

            x = gen_popops_ops.ipu_multi_slice(sliced_table,
                                               adjusted_indices,
                                               name=name)

            # Mask out any values which are not in range.
            mask = array_ops.expand_dims(mask, 1)
            mask = math_ops.cast(mask, x.dtype)
            return x * mask

        # Create the first lookup.
        table_sliced = array_ops.slice(table, [0, 0],
                                       [split_size, table_shape[1]])
        output = func(table_sliced, indices, 0)

        for i in range(1, serialization_factor):
          min_idx = split_size * i

          table_sliced = array_ops.slice(table, [min_idx, 0],
                                         [split_size, table_shape[1]])
          masked_output = func(table_sliced, indices, min_idx)
          # Add the masked slice
          output = math_ops.add(output, masked_output, name=f"slice_{i}")

        # Need to redefine the gradient function.
        def grad(*dy):
          return [
              gen_popops_ops.ipu_multi_update_add(array_ops.zeros_like(table),
                                                  indices=indices,
                                                  updates=dy[0],
                                                  scale=array_ops.constant(
                                                      1, table.dtype)), None
          ]

        return output, grad

      result = serialized_embedding_lookup(params_2d, ids_flat)

  # Reshape into [ids[0], ... , ids[n - 1], params[1], ..., params[n - 1]]
  return array_ops.reshape(result, list(ids_shape) + list(params_shape))


class HostEmbeddingOptimizerSpec:
  # There are unused arguments because we are also defining an interface that will
  # be used by subclasses.
  # pylint: disable=W0613
  """ Description of the Host Embedding optimizer.

      Despite the embedding living on the host, we want to compute the gradients
      on the device. Additionally, the communication channel between the device
      and host is opaque to TensorFlow. For these reasons we need to describe
      the optimizer parameters separately.

      Currently only supports SGD.

  """
  def __init__(self, learning_rate, optimizer_name=None):
    """
    Create a HostEmbeddingOptimizerSpec.

    Args:
        learning_rate: The SGD learning rate.

    """
    self._learning_rate = learning_rate
    if optimizer_name is None:
      optimizer_name = "SGD"
    self._optimizer_name = optimizer_name

  def get_learning_rate(self):
    """
    Get the optimizer learning rate.

    Returns:
      The learning rate.

    """
    return self._learning_rate

  def create_lookup_instruction(self, embedding_tensor, indices, slot_vars,
                                partition_strategy, name):
    """
    Create a lookup instruction.

    This will be called from the `HostEmbedding` wrapper class.

    Args:
        embedding_tensor: The TF embedding tensor bound to the CPU.
        indices: The TF indices tensor bound to the IPU.
        slot_vars: Any created slot variables.
        partition_strategy: The user selected partition strategy.
        name: The name of the host embedding.

    Returns:
      The result of the embedding lookup in an IPU tensor.

    """
    if self.get_learning_rate() == 0:
      return gen_pop_datastream_ops.ipu_device_embedding_lookup(
          indices,
          embedding_id=name,
          embedding_shape=embedding_tensor.shape,
          dtype=embedding_tensor.dtype,
          partition_strategy=partition_strategy)

    with variable_scope.variable_scope(name, reuse=variable_scope.AUTO_REUSE):
      dummy = variable_scope.get_variable("__dummy",
                                          shape=[],
                                          dtype=embedding_tensor.dtype,
                                          trainable=True)
    return gen_pop_datastream_ops.ipu_device_embedding_lookup_trainable(
        dummy,
        indices,
        embedding_id=name,
        embedding_shape=embedding_tensor.shape,
        optimizer=self._optimizer_name,
        partition_strategy=partition_strategy,
        learning_rate=self.get_learning_rate())

  def create_register_instruction(self, embedding_tensor, slot_vars, name):
    """
    Create a register instruction.

    This will be called when entering the `HostEmbedding` context manager.

    Args:
        embedding_tensor: The TF embedding tensor bound to the CPU.
        slot_vars: Any created slot variables.
        name: The name of the host embedding.

    Returns:
      The register instruction.

    """
    return gen_pop_datastream_ops.ipu_host_embedding_register(
        embedding_tensor, name, optimizer=self._optimizer_name)

  def create_deregister_instruction(self, embedding_tensor, slot_vars, name):
    """
    Create a deregister instruction.

    This will be called when exiting the `HostEmbedding` context manager.

    Args:
        embedding_tensor: The TF embedding tensor bound to the CPU.
        slot_vars: Any created slot variables.
        name: The name of the host embedding.

    Returns:
      The deregister instruction.

    """
    return gen_pop_datastream_ops.ipu_host_embedding_deregister(
        embedding_tensor, name)

  def create_slot_variables(self, embedding_tensor, name):
    """
    Create any required slot variables for this optimiser.

    This will be called when exiting the `HostEmbedding` context manager.

    Args:
        embedding_tensor: The TF embedding tensor bound to the CPU.
        name: The name of the host embedding.

    Returns:
      A list of TF tensors bound to the CPU.

    """
    return []


class HostEmbeddingSGDGAOptimizerSpec(HostEmbeddingOptimizerSpec):
  """ Description of the Host Embedding optimizer that uses SGD and
      gradient accumulation.

  """
  def __init__(self, learning_rate, accumulation_factor):
    """
    Create a HostEmbeddingSGDGAOptimizerSpec.

    Args:
        learning_rate: The SGD learning rate.
        accumulation_factor: The gradient accumulation factor (number of
          mini-batches the gradients will be accumulated for).

    """
    if accumulation_factor > 1:
      super().__init__(learning_rate, "SGD+GA")
    else:
      super().__init__(learning_rate)

    self._accumulation_factor = accumulation_factor

  def get_accumulation_factor(self):
    """
    Get the optimizer gradient accumulation factor.

    Returns:
      The gradient accumulation factor.

    """
    return self._accumulation_factor


class HostEmbedding:
  """ Host Embedding wrapper.

      HostEmbedding encapsulates the embedding tensor and the additional
      meta-data required to coordinate the host embedding and the device lookup.
      Through an instance of this class, an IPU can perform lookups on an
      embedding that resides on the host.

      It is assumed that the given embedding will be rank two where the
      outermost dimension (dimension zero) is the token dimension, and the
      innermost dimension is the encoding dimension.

  """
  def __init__(self,
               name,
               embedding_tensor,
               partition_strategy="TOKEN",
               optimizer_spec=None):
    """
    Create a HostEmbedding.

    Args:
        name: The name which uniquely identifies the embedding.
        embedding_tensor: The tensor which holds the embedding.
        optimizer_spec: A description of how the embedding will be optimized.
            When `None`, the embedding is assumed to not be trainable.
    """
    if not tensor_util.is_tensor(embedding_tensor):
      raise ValueError(
          "HostEmbedding embedding_tensor is not a tensorflow tensor")

    if not isinstance(optimizer_spec,
                      (type(None), HostEmbeddingOptimizerSpec)):
      raise ValueError(
          "HostEmbedding optimizer_spec is not a HostEmbeddingOptimizerSpec" +
          " or None")

    if optimizer_spec is None:
      optimizer_spec = HostEmbeddingOptimizerSpec(0)

    if partition_strategy not in ["TOKEN", "ENCODING"]:
      raise ValueError("Unknown partition strategy " + str(partition_strategy))

    self._name = name
    self._embedding_tensor = embedding_tensor
    self._partition_strategy = partition_strategy
    self._optimizer_spec = optimizer_spec
    self._has_lookup = False
    self._slot_vars = optimizer_spec.create_slot_variables(
        self._embedding_tensor, self._name)

  def get_embedding_tensor(self):
    """ Retrieve the CPU bound embedding tensor.

        Returns:
          The TF CPU tensor for the embedding.
    """
    return self._embedding_tensor

  def register(self, session=None):
    """ Creates a host embedding context manager bound to the given session.

        Args:
          session: The session to register the embedding to.
        Returns:
          A Python context manager object. This object manages the lifetime
          of the host embedding connection to the IPU.
    """

    if (session is None) and (not context.executing_eagerly()):
      raise ValueError(
          "HostEmbedding.register requires a session when eager execution"
          "is disabled.")

    # Define this class within the function scope to make it inaccessible to
    # the user.
    class HostEmbeddingScope:
      # We use protected access as though `HostEmbeddingScope` is a friend
      # class of `HostEmbedding`.
      # pylint: disable=W0212
      def __init__(self, parent, session=None):
        self._parent = parent
        self._session = session

      def _register(self):
        return self._parent._optimizer_spec.create_register_instruction(
            self._parent._embedding_tensor, self._parent._slot_vars,
            self._parent._name)

      def _deregister(self):
        return self._parent._optimizer_spec.create_deregister_instruction(
            self._parent._embedding_tensor, self._parent._slot_vars,
            self._parent._name)

      def __enter__(self):
        if self._session is not None:
          self._session.run(self._register())
        else:
          self._register()
        return self._parent

      def __exit__(self, exception_type, exception_value, traceback):
        if self._session is not None:
          self._session.run(self._deregister())
        else:
          self._deregister()

    return HostEmbeddingScope(self, session)

  def __call__(self, *args, **kwargs):
    # Keeping the old function just so an exception can be used to inform
    # users of the API change.
    raise NotImplementedError(
        "HostEmbedding.__call__ is not supported. "
        "Please use the context manager created with HostEmbedding.register.")

  def lookup(self, indices, clip_indices=True):
    """ Perform a host embedding lookup on an IPU.

        Args:
          indices: The indices to lookup.
          clip_indices: Whether to enforce a valid range on the lookup
                        indices with clipping. When False, out-of-range values
                        have undefined behaviour.
        Returns:
          A Tensor containing the elements requested by the user indices.
    """
    indices_shape = indices.shape.as_list()

    # Optionally clip the indices to a safe range
    if clip_indices:
      indices = clip_ops.clip_by_value(indices, 0,
                                       self._embedding_tensor.shape[0] - 1)

    # Flatten the indices.
    num_indices = reduce(mul, indices_shape, 1)
    indices_flat = array_ops.reshape(indices, [num_indices])

    result = self._optimizer_spec.create_lookup_instruction(
        self._embedding_tensor, indices_flat, self._slot_vars,
        self._partition_strategy, self._name)

    self._has_lookup = True
    # Reshape the result back to the caller's expected shape
    return array_ops.reshape(
        result,
        list(indices_shape) + [self._embedding_tensor.shape[1]])


def create_host_embedding(name,
                          shape,
                          dtype,
                          partition_strategy="TOKEN",
                          optimizer_spec=None,
                          initializer=None):
  """ Create a HostEmbedding.

      Args:
        name: The name which uniquely identifies the embedding.
        shape: The shape for the tensor which will hold the embedding.
        dtype: The dtype for the tensor which will hold the embedding.
        partition_strategy: When the IPU system is configured with an IPUConfig
          instance that has its `experimental.enable_remote_buffer_embedding`
          option set to `True`, and when using
          replication, the embedding must be distributed across the replicas.
          This option decides on which axis the embedding will be split. Options
          are "TOKEN" or "ENCODING".
        optimizer_spec: A description of how the embedding will be optimized.
          When `None`, the embedding is assumed to not be trainable.
        initializer: The initializer to use when creating the embedding tensor.

      Returns:
        A `HostEmbedding` object that wraps the created embedding tensor.

  """
  with ops.device('cpu'):
    embedding_tensor = variable_scope.get_variable(name,
                                                   shape=shape,
                                                   dtype=dtype,
                                                   initializer=initializer,
                                                   use_resource=False)
  return HostEmbedding(name,
                       embedding_tensor,
                       partition_strategy=partition_strategy,
                       optimizer_spec=optimizer_spec)
