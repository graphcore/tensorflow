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
Popops all to all and all gather operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops


def all_to_all(x,
               split_dimension,
               concat_dimension,
               replication_factor,
               name=None):
  """ Perform an XLA all to all operation across all replicas.
    (See https://www.tensorflow.org/xla/operation_semantics#alltoall)

    Args:
      split_dimension: A value in the interval [0,n) that names the dimension
                      along which the operand is split
      concat_dimension: A value in the interval [0,n) that names the dimension
                      along which the split blocks are concatenated.
      replication_factor: The replication factor of the model.
      name: Optional op name.
    Returns:
      A tensor of the same size where each replica will have a different value.
    """

  return gen_popops_ops.ipu_all_to_all(x,
                                       split_dimension=split_dimension,
                                       concat_dimension=concat_dimension,
                                       number_of_replicas=replication_factor,
                                       name=name)


def all_gather(x, replication_factor, name=None):
  """ Gather the data on all replicas to all other replicas. Each replica will
    have the exact same output.

    Args:
      x: The tensor to gather
      replication_factor: The number of replicas in each collective group.
        If less than the total number of replicas in the model, the replicas
        are divided into consecutive groups of the given size, and the
        collective operation is performed within each respective group.
        If there are `N` total replicas denoted `{0, ... N-1}` and
        `replication_factor` is `k`, then the groups are:
        `{0, 1, ... k-1}, {k, ... 2k-1} ... {N-k-1, ... N-1}`.
        Note that `N` must be evenly divisible by `k`, otherwise an exception
        will be thrown during compilation.
      name: Optional op name.

    Returns:
      A tensor of [replication_factor][x] with each replica in the same group
      having the same tensor.
    """
  return gen_popops_ops.ipu_all_gather(x,
                                       replication_factor=replication_factor,
                                       name=name)
