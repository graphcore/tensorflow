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
from tensorflow.python.ipu import functional_ops_grad
from tensorflow.python.ops import control_flow_util_v2 as util


@ops.RegisterGradient("IpuGelu")
def _ipu_gelu_grad(op, grad):
  """Gradients for the IpuGelu op."""
  x = op.inputs[0]
  return [gen_popnn_ops.ipu_gelu_grad(grad, x)]


@ops.RegisterGradient("MultiConv")
def _multi_conv_grad(op, *grads):
  """The gradient of a MultiConv op."""
  func_grad_graph, func_grad_inputs = \
    functional_ops_grad._get_gradients_for_function(op, *grads) # pylint: disable=protected-access
  outputs = gen_functional_ops.multi_conv(
      func_grad_inputs,
      to_apply=util.create_new_tf_function(func_grad_graph),
      Tout=func_grad_graph.output_types,
      output_shapes=func_grad_graph.output_shapes)

  return func_graph_module.pack_sequence_as(func_grad_graph.structured_outputs,
                                            outputs)
