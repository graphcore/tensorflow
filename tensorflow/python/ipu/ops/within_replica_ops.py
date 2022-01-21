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
"""
Popops within replica operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops

from tensorflow.python.ops import array_ops


def reduce_scatter(input_shards, op):
  """
  Reduce the given sharded tensors with the results scattered across the
  shards. If the tensors contain fewer/more elements than shards then the
  results will be 0 padded. Example:

  .. code-block:: none

    Input: IPU0 [x0, y0, z0]
           IPU1 [x1, y1, z1]
           IPU2 [x2, y2, z2]
           IPU3 [x3, y3, z3]

    Input: IPU0 [0]
           IPU1 [op(y0, y1, y2)]
           IPU2 [op(z0, z1, z2)]
           IPU3 [op(x0, x1, x2)]

  Args:
    input_shards: The tensors to reduce. These are expected to be supplied in
      increasing shard order, so that input_shards[0] is on shard0 and
      input_shard[i] is on shard i. Additionally these tensors must be of the
      same type and of rank 0 or 1.
    op: Reduce operation, valid ops are: COLLECTIVE_OP_ADD,
      COLLECTIVE_OP_MUL, COLLECTIVE_OP_MIN, COLLECTIVE_OP_MAX,
      COLLECTIVE_OP_LOGICAL_AND, COLLECTIVE_OP_LOGICAL_OR,
      COLLECTIVE_OP_LOCAL.

  Returns:
    A tuple of tensors, where each tensor contains 0 or more reduction results.
    Element i is the `Tensor` mapped to shard i.
  """
  _validate_inputs(input_shards)

  input_shards = _reshape_scalars(input_shards)
  return gen_popops_ops.ipu_reduce_scatter_within_replica(
      _pad_to_equal_size(input_shards), collective_op=op)


def all_gather(input_shards, axis=0):
  """
  Perform an all gather for a list of sharded tensors within a replica.

  Args:
    input_shards: the sharded input tensors to gather. These are expected to
      be supplied in incrementing sharded order, so that input_shards[0] is on
      shard0 and input_shard[i] is on shard i. Additionally these tensors must
      all be of the same type and of the same rank.
:
  Returns:
    A tuple of tensors that contains a copy of the data for each shard. Element
    i is the `Tensor` mapped to shard i. Each sub tensor is of shape
    `tf.concat(input_shards, axis=axis)`.
  """
  _validate_inputs(input_shards)

  flattened_inputs = _flatten_tensors(input_shards)
  gathered = gen_popops_ops.ipu_all_gather_within_replica(flattened_inputs)

  # Inputs have been validated to have the same rank, so just check the first.
  scalar_input = input_shards[0].shape.rank == 0
  if not scalar_input:
    shard_shape = _concatenate_shapes(input_shards, axis)
    return [array_ops.reshape(shard, shard_shape) for shard in gathered]

  return gathered


def _validate_inputs(input_shards):
  """
  Raises an exception if the given input tensors are not valid for a
  within_replica op.The tensors must be unique, and of the same type and rank.
  """
  unique_inputs = set(shard.ref() for shard in input_shards)
  if len(unique_inputs) != len(input_shards):
    raise ValueError("input_shards should not contain any duplicate entries.")

  input_types = [shard.dtype for shard in input_shards]
  unique_types = set(input_types)
  if len(unique_types) != 1:
    raise ValueError("Expected tensors to all be of the same type.")

  ranks = [shard.shape.rank for shard in input_shards]
  unique_ranks = set(ranks)
  if len(unique_ranks) != 1:
    raise ValueError("Expected tensors to all be of the same rank.")


def _reshape_scalars(tensors):
  """
  Reshape any scalar tensors to be rank 1.
  """
  for i, tensor in enumerate(tensors):
    if tensor.shape.rank == 0:
      tensors[i] = array_ops.reshape(tensor, [1])
  return tensors


def _pad_to_equal_size(flat_tensors):
  """
  Zero pad the given tensors so they have same size as the largest
  tensor in flat_tensors.
  """
  equal_size_tensors = []

  elements_needed = max(tensor.shape[0] for tensor in flat_tensors)
  for tensor in flat_tensors:
    tensor_size = tensor.shape[0]

    if tensor_size != elements_needed:
      assert tensor_size < elements_needed
      padded = array_ops.pad(tensor,
                             paddings=[[0, elements_needed - tensor_size]])
      equal_size_tensors.append(padded)
    else:
      equal_size_tensors.append(tensor)

  return equal_size_tensors


def _concatenate_shapes(tensors, axis):
  """ Utility for generating a concatenated tensor shape, where elements are
    joined across the given axis. Expects tensor dimensions to be the same
    outside of this axis. """
  concatenated_shape = tensors[0].shape.as_list()
  assert axis < len(concatenated_shape)

  for i, shard in enumerate(tensors[1:]):
    shard_shape = shard.shape.as_list()

    remaining_dims_match = concatenated_shape[:axis] == shard_shape[:axis] and\
                       concatenated_shape[axis + 1:] == shard_shape[axis + 1:]
    if remaining_dims_match:
      concatenated_shape[axis] += shard_shape[axis]
    else:
      raise ValueError(
          f"Expecting tensor dimensions to all have the same sizes, except for "
          f"dim {axis}, but got shapes {concatenated_shape} and {shard_shape} "
          f"for tensors 0 and {i+1}")

  return concatenated_shape


def _flatten_tensors(tensors):
  """ Flatten the tensors in the given list so they are of rank 1. """
  flattened_inputs = []
  for tensor in tensors:
    if tensor.shape.rank != 1:
      flattened_inputs.append(array_ops.reshape(tensor, [-1]))
    else:
      flattened_inputs.append(tensor)

  return flattened_inputs
