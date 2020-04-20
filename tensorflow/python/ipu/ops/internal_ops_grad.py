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


@ops.RegisterGradient("IpuUserReadWriteOp")
def _poputil_cpu_user_operation_layer_backward(op, *grads):
  library_path = op.get_attr("library_path").decode("utf-8")
  op_name = op.get_attr("op_name").decode("utf-8")

  separate_grads = op.get_attr("separate_gradients")

  result = []
  if separate_grads:
    for t in enumerate(op.inputs):
      outs = {
          "output_types": [t[1].dtype],
          "output_shapes": [t[1].shape],
      }
      o = gen_poputil_ops.ipu_user_read_write_op(list(grads) +
                                                 list(op.outputs) +
                                                 list(op.inputs),
                                                 library_path=library_path,
                                                 op_name=op_name + "_grad",
                                                 name=op.name + "_grad",
                                                 separate_gradients=True,
                                                 is_gradient=True,
                                                 partial_derivative_index=t[0],
                                                 **outs)[0]
      result.append(o)
  else:
    outs = {
        "output_types": [t.dtype for t in op.inputs],
        "output_shapes": [t.shape for t in op.inputs],
    }

    result = gen_poputil_ops.ipu_user_read_write_op(list(grads) +
                                                    list(op.outputs) +
                                                    list(op.inputs),
                                                    library_path=library_path,
                                                    op_name=op_name + "_grad",
                                                    name=op.name + "_grad",
                                                    separate_gradients=False,
                                                    is_gradient=True,
                                                    partial_derivative_index=0,
                                                    **outs)

  return result


@ops.RegisterGradient("IpuUserOp")
def _poputil_precompiled_user_op_layer_backward(op, *grads):
  library_path = op.get_attr("library_path").decode("utf-8")
  op_name = op.get_attr("op_name").decode("utf-8")
  gp_path = op.get_attr("gp_path").decode("utf-8")

  separate_grads = op.get_attr("separate_gradients")

  result = []
  if separate_grads:
    for t in enumerate(op.inputs):
      outs = {
          "output_types": [t[1].dtype],
          "output_shapes": [t[1].shape],
      }
      o = gen_poputil_ops.ipu_user_op(list(grads) + list(op.outputs) +
                                      list(op.inputs),
                                      library_path=library_path,
                                      op_name=op_name + "_grad",
                                      gp_path=gp_path,
                                      name=op.name + "_grad",
                                      separate_gradients=True,
                                      is_gradient=True,
                                      partial_derivative_index=t[0],
                                      **outs)[0]
      result.append(o)
  else:
    outs = {
        "output_types": [t.dtype for t in op.inputs],
        "output_shapes": [t.shape for t in op.inputs],
    }

    result = gen_poputil_ops.ipu_user_op(list(grads) + list(op.outputs) +
                                         list(op.inputs),
                                         library_path=library_path,
                                         op_name=op_name + "_grad",
                                         gp_path=gp_path,
                                         name=op.name + "_grad",
                                         separate_gradients=False,
                                         is_gradient=True,
                                         partial_derivative_index=0,
                                         **outs)

  return result
