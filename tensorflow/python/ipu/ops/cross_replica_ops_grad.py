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
"""Gradients for popops cross replica operators."""

from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops
from tensorflow.python.framework import ops


@ops.RegisterGradient("IpuCrossReplicaSum")
def _cross_replica_sum_grad(op, grads):
  """Gradients for the IpuCrossReplicaSum op."""
  return [
      gen_popops_ops.ipu_cross_replica_sum(
          grads, replica_group_size=op.get_attr("replica_group_size"))
  ]
