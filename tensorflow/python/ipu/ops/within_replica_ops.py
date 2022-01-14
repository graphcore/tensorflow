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


def all_gather(input_shards, axis=0):
  """
  Perform an all gather for a list of sharded tensors within a replica.

  Args:
    input_shards: the sharded input tensors to gather. These are expected to
    be supplied in incrementing sharded order, so that input_shards[0] is on
    shard0 and input_shard[i] is on shard i. Additionally these tensors must
    all be of the same type and of the same rank.

  Returns:
    A `Tensor` that contains a copy of the data for each shard. Index i
    in the outermost dimension is mapped to shard i. Each sub tensor is of
    shape `tf.concat(input_shards, axis=axis)`.
  """
  unique_inputs = set(shard.ref() for shard in input_shards)
  if len(unique_inputs) != len(input_shards):
    raise ValueError("input_shards should not contain any duplicated entries")

  input_types = [shard.dtype for shard in input_shards]
  unique_types = set(input_types)
  if len(unique_types) != 1:
    raise ValueError("Expected tensors to all be of the same type.")

  ranks = [len(shard.shape) for shard in input_shards]
  unique_ranks = set(ranks)
  if len(unique_ranks) != 1:
    raise ValueError("Expected tensors to all be of the same rank.")

  flattened_inputs = _flatten_tensors(input_shards)
  gathered = gen_popops_ops.ipu_all_gather_within_replica(flattened_inputs)

  scalar_input = 0 in unique_ranks
  if not scalar_input:
    shard_shape = _concatenate_shapes(input_shards, axis)
    return [array_ops.reshape(shard, shard_shape) for shard in gathered]

  return gathered


def _concatenate_shapes(tensors, axis):
  """ Utility for generating a concatenated tensor shape, where elements are
    joined across the given axis. Expects tensor dimensions to be the same
    outside of this axis. """
  concatenated_shape = tensors[0].shape.as_list()

  for i, shard in enumerate(tensors[1:]):
    shard_shape = shard.shape.as_list()

    remaining_dims_match = concatenated_shape[:axis] == shard_shape[:axis] and\
                       concatenated_shape[axis + 1:] == shard_shape[axis + 1:]
    if remaining_dims_match:
      concatenated_shape[axis] += shard.shape[axis]
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
    rank = len(tensor.shape)
    if rank != 1:
      flattened_inputs.append(array_ops.reshape(tensor, [-1]))
    else:
      flattened_inputs.append(tensor)

  return flattened_inputs
