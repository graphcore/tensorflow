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
Popops reduce scatter operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.util import nest
from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops


def reduce_scatter(x, replication_factor, op='COLLECTIVE_OP_ADD', name=None):
  """Reduce the given replicated tensor with the result scattered across
  the replicas. For an input of shape `[num_elements]`, the output will have
  shape `[ceil(num_elements / replication_factor)]`. If `replication_factor`
  does not evenly divide `num_elements`, the result is zero-padded. Example:

  .. code-block:: none

    Input:  Replica0: [x0, y0, z0]
            Replica1: [x1, y1, z1]
    Output: Replica0: [x0 + x1, y0 + y1]
            Replica1: [z0 + z1, 0]

  Args:
    x: The input tensor or list of tensors. The tensors must have rank 1.
    replication_factor: The number of replicas in each collective group.
      If less than the total number of replicas in the model, the replicas
      are divided into consecutive groups of the given size, and the
      collective operation is performed within each respective group.
      If there are `N` total replicas denoted `{0, ... N-1}` and
      `replication_factor` is `k`, then the groups are:
      `{0, 1, ... k-1}, {k, ... 2k-1} ... {N-k-1, ... N-1}`.
      Note that `N` must be evenly divisible by `k`, otherwise an exception
      will be thrown during compilation.
    op: Reduce operation, valid ops are: COLLECTIVE_OP_ADD,
      COLLECTIVE_OP_MUL, COLLECTIVE_OP_MIN, COLLECTIVE_OP_MAX,
      COLLECTIVE_OP_LOGICAL_AND, COLLECTIVE_OP_LOGICAL_OR,
      COLLECTIVE_OP_SQUARE_ADD, COLLECTIVE_OP_LOCAL and
      COLLECTIVE_OP_MEAN.
    name: Optional op name.

  Returns:
    A `Tensor` or list of `Tensor`s. The shape of each output will be
    `[ceil(input_length / number_of_replicas)]`.
  """

  flat_x = nest.flatten(x)
  output = gen_popops_ops.ipu_reduce_scatter(
      flat_x, replication_factor=replication_factor, op=op, name=name)
  if len(output) == 1:
    return output[0]
  return output
