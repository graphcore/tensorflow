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
"""Gradients for Popnn operators."""

from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.framework import ops
"""
    These gradient function should *never* be called directly.
"""
@ops.RegisterGradient("IpuRemap")
def _poputil_remap_layer_backward(op, grads):
  """Gradients for the IpuRemap op."""
  return grads


@ops.RegisterGradient("IpuRemapDeduce")
def _poputil_remap_deduce_layer_backward(op, grads):
  """Gradients for the IpuRemapDeduce op."""
  return grads


@ops.RegisterGradient("IpuPrintTensor")
def _poputil_print_tensor_layer_backward(op, grads):
  """Gradients for the IpuPrintTensor op."""
  return grads


@ops.RegisterGradient("IpuFifo")
def _poputil_fifo_backward(op, grads):
  """Gradients for the IpuFifo op."""
  return [None] * len(grads)


@ops.RegisterGradient("IpuSuggestRecompute")
def _poputil_recompute_backward(op, grads):
  """Gradients for the IpuSuggestRecompute op."""
  return grads


@ops.RegisterGradient("IpuBlockRecompute")
def _poputil_block_recompute_backward(op, grads):
  """Gradients for the IpuBlockRecompute op."""
  return grads


def _filter_inputs(inputs, inputs_with_gradients):
  inputs_with_gradients = set(inputs_with_gradients)
  return [
      t if i in inputs_with_gradients else None for i, t in enumerate(inputs)
  ]


def _expand_result(outputs, n, inputs_with_gradients):
  result = []
  for i in xrange(n):
    if i in inputs_with_gradients:
      result.append(outputs.pop(0))
    else:
      result.append(None)
  assert not outputs and len(result) == n
  return result


def _poputil_op_layer_backward(op, grads, add_op):
  separate_grads = op.get_attr("separate_gradients")
  inputs_with_gradients = op.get_attr("inputs_with_gradients")

  result = []
  layout = list(grads) + list(op.outputs) + list(op.inputs)
  inputs = _filter_inputs(op.inputs, inputs_with_gradients)
  if separate_grads:
    for op_input_index, op_input in enumerate(inputs):
      if op_input is not None:
        result.append(
            add_op(layout, len(grads), True, op_input_index, [0],
                   [op_input.dtype], [op_input.shape])[0])
  else:
    result = add_op(layout, len(grads), False, 0, inputs_with_gradients,
                    [t.dtype for t in inputs if t is not None],
                    [t.shape for t in inputs if t is not None])

  return _expand_result(result, len(op.inputs), inputs_with_gradients)


@ops.RegisterGradient("IpuUserReadWriteOp")
def _poputil_cpu_user_operation_layer_backward(op, *grads):
  library_path = op.get_attr("library_path").decode("utf-8")
  op_name = op.get_attr("op_name").decode("utf-8") + "_grad"
  gradient_attributes = op.get_attr("gradient_attributes")

  def add_op(layout, gradient_size, separate_gradients, op_input_index,
             inputs_with_gradients, output_types, output_shapes):
    return gen_poputil_ops.ipu_user_read_write_op(
        layout,
        library_path=library_path,
        op_name=op_name,
        separate_gradients=separate_gradients,
        gradient_size=gradient_size,
        partial_derivative_index=op_input_index,
        inputs_with_gradients=inputs_with_gradients,
        attributes=gradient_attributes,
        output_types=output_types,
        output_shapes=output_shapes)

  return _poputil_op_layer_backward(op, grads, add_op)


@ops.RegisterGradient("IpuUserOp")
def _poputil_precompiled_user_op_layer_backward(op, *grads):
  library_path = op.get_attr("library_path").decode("utf-8")
  op_name = op.get_attr("op_name").decode("utf-8") + "_grad"
  gp_path = op.get_attr("gp_path").decode("utf-8")
  gradient_attributes = op.get_attr("gradient_attributes")

  def add_op(layout, gradient_size, separate_gradients, op_input_index,
             inputs_with_gradients, output_types, output_shapes):
    return gen_poputil_ops.ipu_user_op(
        layout,
        library_path=library_path,
        op_name=op_name,
        gp_path=gp_path,
        separate_gradients=separate_gradients,
        gradient_size=gradient_size,
        partial_derivative_index=op_input_index,
        inputs_with_gradients=inputs_with_gradients,
        attributes=gradient_attributes,
        output_types=output_types,
        output_shapes=output_shapes)

  return _poputil_op_layer_backward(op, grads, add_op)
