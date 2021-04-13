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
