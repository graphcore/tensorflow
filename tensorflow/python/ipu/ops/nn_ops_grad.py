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
"""Gradients for Popnn operators."""

from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from tensorflow.compiler.plugin.poplar.ops import gen_functional_ops
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ipu import functional_ops
from tensorflow.python.ipu import functional_ops_grad
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops.nn_grad import _BroadcastMul


@ops.RegisterGradient("IpuGelu")
def _ipu_gelu_grad(op, grad):
  """Gradients for the IpuGelu op."""
  x = op.inputs[0]
  return [gen_popnn_ops.ipu_gelu_grad(grad, x)]


@ops.RegisterGradient("IpuHardSigmoid")
def _ipu_hard_sigmoid_grad(op, grad):
  """Gradients for the IpuHardSigmoid op."""
  x = op.inputs[0]
  return [gen_popnn_ops.ipu_hard_sigmoid_grad(grad, x)]


@ops.RegisterGradient("IpuSwish")
def _ipu_swish_grad(op, grad):
  """Gradients for the IpuSwish op."""
  x = op.inputs[0]
  return [gen_popnn_ops.ipu_swish_grad(grad, x)]


@ops.RegisterGradient("IpuSoftmax")
def _ipu_softmax_grad(op, grad):
  """Gradients for the IpuSoftmax op."""
  # borrowed from tensorflow/python/ops/nn_grad.py
  softmax = op.outputs[0]
  sum_channels = math_ops.reduce_sum(grad * softmax, -1, keepdims=True)
  return (grad - sum_channels) * softmax


@ops.RegisterGradient("IpuStableSoftmax")
def _ipu_stable_softmax_grad(op, grad):
  """Gradients for the IpuSoftmax op."""
  return _ipu_softmax_grad(op, grad)


@ops.RegisterGradient("MultiConv")
def _multi_conv_grad(op, *grads):
  """The gradient of a MultiConv op."""
  func_grad_graph, func_grad_inputs, constant_outputs, \
    grads_written_as_outputs = \
    functional_ops_grad._get_gradients_for_function(op, *grads) # pylint: disable=protected-access
  outputs = gen_functional_ops.multi_conv(
      func_grad_inputs,
      to_apply=util.create_new_tf_function(func_grad_graph),
      Tout=func_grad_graph.output_types,
      output_shapes=func_grad_graph.output_shapes,
      option_flags=op.get_attr("option_flags"))

  outputs = functional_ops._replace_outputs(outputs, constant_outputs)  # pylint: disable=protected-access
  outputs = functional_ops_grad._extract_and_replace_captured_grads(
      grads_written_as_outputs, outputs)
  return functional_ops._pack_sequence_as(  # pylint: disable=protected-access
      func_grad_graph.structured_outputs, outputs)


@ops.RegisterGradient("PopnnCTCLossWithLogits")
@ops.RegisterGradient("PopnnCTCLossWithLogProbs")
def _ctc_loss_grad(op, loss_grad, _):
  """The gradient of PopnnCTCLossWithLogits and PopnnCTCLossWithLogProbs ops."""
  op_grad = array_ops.prevent_gradient(
      op.outputs[1],
      message="Second order derivative is not currently available for CTC Loss."
  )

  return [_BroadcastMul(loss_grad, op_grad), None, None, None]
