# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
Popops within replica operators gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.ipu.ops import within_replica_ops
from tensorflow.python.framework import ops


@ops.RegisterGradient("IpuAllGatherWithinReplica")
def _ipu_all_gather_within_replica_grad(op, *grads):  #pylint: disable=unused-argument
  return within_replica_ops.reduce_scatter(grads, op="COLLECTIVE_OP_ADD")


@ops.RegisterGradient("IpuReduceScatterWithinReplica")
def _ipu_reduce_scatter_within_replica_grad(op, *grads):  #pylint: disable=unused-argument
  return within_replica_ops.all_gather(grads)


@ops.RegisterGradient("IpuAllReduceWithinReplica")
def _ipu_all_reduce_within_replica_grad(op, *grads):
  original_op = op.get_attr("collective_op")
  return within_replica_ops.all_reduce(grads, op=original_op)
