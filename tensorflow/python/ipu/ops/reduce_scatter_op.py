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

from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops


def reduce_scatter(x, replication_factor, name=None):
  """Reduce (sum) the given replicated tensor with the result scattered across
  the replicas. For an input of shape `[num_elements]`, the output will have
  shape `[ceil(num_elements / replication_factor)]`. If `replication_factor`
  does not evenly divide `num_elements`, the result is zero-padded. Example:

  .. code-block:: none

    Input:  Replica0: [x0, y0, z0]
            Replica1: [x1, y1, z1]
    Output: Replica0: [x0 + x1, y0 + y1]
            Replica1: [z0 + z1, 0]

  Args:
    x: The input `Tensor`. Must have rank 1.
    replication_factor: The replication factor of the model.
    name: Optional op name.

  Returns:
    A `Tensor` with the result for this replica.
  """
  return gen_popops_ops.ipu_reduce_scatter(
      x, replication_factor=replication_factor, name=name)
