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


def all_gather(x, replication_factor, name):
  """ Gather the data on all replicas to all other replicas. Each replica will
    have the exact same output.

    Args:
      x: The tensor to gather
      replication_factor: The replication factor of the model.
      name: Optional op name.

    Returns:
      A tensor of [num_replicas][x] with each replica having the same tensor.
    """
  return gen_popops_ops.ipu_all_gather(x,
                                       replication_factor=replication_factor,
                                       name=name)
