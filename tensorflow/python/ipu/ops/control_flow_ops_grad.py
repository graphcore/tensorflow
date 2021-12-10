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
"""Gradients for control flow operations."""

from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.framework import ops


@ops.RegisterGradient("IpuBarrier")
def _barrier_grad(op, *grads):
  """Gradients for the IpuBarrier op."""
  return gen_poputil_ops.ipu_barrier(grads) if op.get_attr(
      "insert_barrier_for_gradients") else grads
