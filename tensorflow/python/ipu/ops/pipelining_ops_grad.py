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
"""Gradients for Pipelining operators."""

from tensorflow.compiler.plugin.poplar.ops import gen_functional_ops
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ipu import functional_ops
from tensorflow.python.ipu import functional_ops_grad
from tensorflow.python.ops import control_flow_util_v2 as util


@ops.RegisterGradient("PipelineStage")
def _pipeline_stage_grad(op, *grads):
  """The gradient of a PipelineStage op."""
  func_grad_graph, func_grad_inputs, constant_outputs = \
      functional_ops_grad._get_gradients_for_function(op, *grads)  # pylint: disable=protected-access
  stage_op = op.outputs[0].op
  stage_id = stage_op.get_attr('stage_id')

  outputs = gen_functional_ops.pipeline_stage_backward(
      func_grad_inputs,
      to_apply=util.create_new_tf_function(func_grad_graph),
      Tout=func_grad_graph.output_types,
      output_shapes=func_grad_graph.output_shapes,
      stage_id=stage_id)

  outputs = functional_ops._replace_outputs(outputs, constant_outputs)  # pylint: disable=protected-access
  return functional_ops._pack_sequence_as(  # pylint: disable=protected-access
      func_grad_graph.structured_outputs, outputs)


@ops.RegisterGradient("Pipeline")
def _pipeline_grad(op, *grads):
  """The gradient of a Pipeline op."""
  raise RuntimeError(
      "Attempting to calculate the gradient of a Pipeline which is not allowed."
      " If you are trying to generate the gradients of operations inside the"
      " pipeline, use the `optimizer_stage` (see"
      " tensorflow.python.ipu.pipelining_ops.pipeline for more information).")


@ops.RegisterGradient("RecomputationCheckpoint")
def _recomputation_checkpoint(_op, grads):
  """Gradients for the RecomputationCheckpoint op."""
  return grads
