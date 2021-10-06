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
"""Gradients for popops embedding operators."""

from tensorflow.compiler.plugin.poplar.ops import gen_popops_ops
from tensorflow.compiler.plugin.poplar.ops import gen_pop_datastream_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


@ops.RegisterGradient("IpuMultiSlice")
def _ipu_multi_update(op, grads):
  """Gradients for the IpuMultiSlice op."""
  return [
      gen_popops_ops.ipu_multi_update_add(
          array_ops.zeros_like(op.inputs[0]),
          indices=op.inputs[1],
          updates=grads,
          scale=array_ops.constant(1, op.inputs[0].dtype),
          indices_are_sorted=op.get_attr("indices_are_sorted")), None
  ]


@ops.RegisterGradient("IpuDeviceEmbeddingLookupTrainable")
def _ipu_host_embedding_lookup_grad(op, grads):
  """Gradients for the IpuDeviceEmbeddingLookupTrainable op."""
  update_op = gen_pop_datastream_ops.ipu_device_embedding_update_add(
      op.outputs[0],
      -grads * op.get_attr("learning_rate"),
      indices=op.inputs[1],
      embedding_id=op.get_attr("embedding_id"),
      embedding_shape=op.get_attr("embedding_shape"),
      partition_strategy=op.get_attr("partition_strategy"))

  if op.get_attr("optimizer") == 'SGD+GA':
    with ops.control_dependencies([update_op]):
      update_op = gen_pop_datastream_ops.ipu_device_embedding_notify(
          embedding_id=op.get_attr("embedding_id"))
  with ops.control_dependencies([update_op]):
    return [array_ops.zeros_like(op.inputs[0]), None]
