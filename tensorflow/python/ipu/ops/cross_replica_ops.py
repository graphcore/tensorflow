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
Popops cross replica operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.ipu.ops import internal_ops

from tensorflow.python.util import nest


def cross_replica_sum(x, replica_group_size=None, name=None):
  """Sum the input tensor across replicas.

  Args:
    x: The local tensor to the sum.
    replica_group_size: The number of replicas in each collective group.
      If None, there is a single group containing all the replicas. If a
      number less than the total number of replicas in the model is
      provided, the replicas are divided into consecutive groups of the
      given size, and the collective operation is performed within each
      respective group. Given `N` total replicas denoted `{0, ... N-1}`
      and a `replica_group_size` of k, the groups are:
      `{0, 1, ... k-1}, {k, ... 2k-1} ... {N-k-1, ... N-1}`.
      Note that `N` must be evenly divisible by `k`, otherwise an exception
      will be thrown during compilation.
    name: Optional op name.

  Returns:
    A `Tensor` which is summed across the replicas in the same group.
  """

  return gen_popops_ops.ipu_cross_replica_sum(
      x, replica_group_size=replica_group_size, name=name)


def cross_replica_mean(x, replica_group_size=None, name=None):
  """Computes the mean of the input tensor across replicas.

  Args:
    x: The local tensor to the mean.
    replica_group_size: The number of replicas in each collective group.
      If None, there is a single group containing all the replicas. If a
      number less than the total number of replicas in the model is
      provided, the replicas are divided into consecutive groups of the
      given size, and the collective operation is performed within each
      respective group. Given `N` total replicas denoted `{0, ... N-1}`
      and a `replica_group_size` of k, the groups are:
      `{0, 1, ... k-1}, {k, ... 2k-1} ... {N-k-1, ... N-1}`.
      Note that `N` must be evenly divisible by `k`, otherwise an exception
      will be thrown during compilation.
    name: Optional op name.

  Returns:
    A `Tensor` which is averaged across the replicas in the same group.
  """

  return gen_popops_ops.ipu_cross_replica_mean(
      x, replica_group_size=replica_group_size, name=name)


def assume_equal_across_replicas(tensors, inplace=False):
  """
  Mark the given tensors as equal across replicas to try and prevent divergent
  control flow compilation errors.

  Divergent control flow describes the situation where program flow differs
  among replicas. This happens when the value of a conditional is not the same
  across all replicas. This is a problem if the conditional body requires a
  cross-replica sync, as only some replicas will reach it. If this happens,
  the execution will hang as the operation waits for all replicas to sync.

  To warn the user about this, Poplar checks for divergent control flow during
  compilation. However since the values of tensors are unknown at compilation
  time it can't be certain whether a tensor will lead to divergent control
  flow or not. `assume_equal_across_replicas` can be used to mark tensors
  which are equal across all replicas and in doing so prevents them causing
  divergency errors, if used in a conditional.

  Args:
    tensors: A tensor or a structure of tensors which will be marked as equal
      across replicas. Note that undefined behaviour will occur if these tensors
      are in fact not equal across replicas.
    inplace: A bool for controlling whether or not the given tensor(s) is copied
      or operated on inplace. This is needed when using
      `assume_equal_across_replicas` with tensor slices.

  Returns:
    A tensor or a structure of tensors which matches shape and type of the
    `tensors` arg. This should be used in place of the args to prevent divergent
    control flow errors.
  """
  inputs = nest.flatten(tensors, expand_composites=True)

  if not inplace:
    # Using IpuAssumeEqualAcrossReplicas with tensor slices requires
    # the whole tensor to be wrapped in an IpuAssumeEqualAcrossReplicas op,
    # which can be difficult/fiddly to get right. Copying provides an easier
    # way to get it working.
    inputs = [internal_ops.remap_deduce(x) for x in inputs]

  outputs = [
      gen_poputil_ops.ipu_assume_equal_across_replicas(inputs=x)
      for x in inputs
  ]
  return nest.pack_sequence_as(tensors, outputs, expand_composites=True)
