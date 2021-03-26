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

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops


@ops.RegisterGradient("IpuSequenceSlice")
def _sequence_slice_grad(op, *grads):
  """Gradients for the IPU SequenceSlice op."""
  src = op.inputs[1]
  num_elems = op.inputs[2]
  src_offsets = op.inputs[3]
  dst_offsets = op.inputs[4]
  zero_unused = op.get_attr("zero_unused")

  return None, gen_popops_ops.ipu_sequence_slice(
      dst=src,
      src=grads[0],
      num_elems=num_elems,
      src_offsets=dst_offsets,
      dst_offsets=src_offsets,
      zero_unused=zero_unused), None, None, None


@ops.RegisterGradient("IpuSequenceSliceUnpack")
def _sequence_slice_unpack_grad(op, *grads):
  """Gradients for the IPU SequenceSliceUnpack op."""
  src = op.inputs[0]
  num_elems = op.inputs[1]
  src_offsets = op.inputs[2]

  grad_src_offsets = math_ops.cumsum(array_ops.concat([[0], num_elems[:-1]],
                                                      0))

  return gen_popops_ops.ipu_sequence_slice(dst=src,
                                           src=grads[0],
                                           num_elems=num_elems,
                                           src_offsets=grad_src_offsets,
                                           dst_offsets=src_offsets,
                                           zero_unused=True), None, None, None
