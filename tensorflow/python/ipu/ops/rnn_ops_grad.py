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

from tensorflow.python.framework import ops
from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
"""
    These gradient function should *never* be called directly.
"""
@ops.RegisterGradient("PopnnLstmLayer")
def _popnn_lstm_layer_backward(op, *grads):
  """Gradients for the PopnnLstmLayer op."""
  if not op.get_attr("is_training"):
    raise ValueError(
        "To use PopnnLstmLayer in gradients, is_training must be set to True.")
  return gen_popnn_ops.popnn_lstm_layer_backprop(
      inputs=op.inputs[0],
      input_h_state=op.inputs[1],
      input_c_state=op.inputs[2],
      kernel=op.inputs[3],
      biases=op.inputs[4],
      output=op.outputs[0],
      output_h_state=op.outputs[1],
      output_c_state=op.outputs[2],
      intermediates=op.outputs[3],
      output_backprop=grads[0],
      output_h_state_backprop=grads[1],
      output_c_state_backprop=grads[2],
      activation=op.get_attr("activation"),
      recurrent_activation=op.get_attr("recurrent_activation"),
      num_channels=op.get_attr("num_channels"),
      partials_dtype=op.get_attr("partials_dtype"),
      is_training=op.get_attr("is_training"),
      available_memory_proportion_bwd=op.get_attr(
          "available_memory_proportion_bwd"),
      options=op.get_attr("options_bwd"),
  )


@ops.RegisterGradient("PopnnDynamicLstmLayer")
def _popnn_dynamic_lstm_layer_backward(op, *grads):
  """Gradients for the PopnnDynamicLstmLayer op."""
  if not op.get_attr("is_training"):
    raise ValueError(
        "To use PopnnDynamicLstmLayer in gradients, is_training must be set to"
        " True.")
  g = gen_popnn_ops.popnn_dynamic_lstm_layer_backprop(
      inputs=op.inputs[0],
      input_h_state=op.inputs[1],
      input_c_state=op.inputs[2],
      kernel=op.inputs[3],
      biases=op.inputs[4],
      seq_len=op.inputs[5],
      output=op.outputs[0],
      output_h_state=op.outputs[1],
      output_c_state=op.outputs[2],
      intermediates=op.outputs[3],
      output_backprop=grads[0],
      output_h_state_backprop=grads[1],
      output_c_state_backprop=grads[2],
      activation=op.get_attr("activation"),
      recurrent_activation=op.get_attr("recurrent_activation"),
      num_channels=op.get_attr("num_channels"),
      partials_dtype=op.get_attr("partials_dtype"),
      is_training=op.get_attr("is_training"),
      available_memory_proportion_bwd=op.get_attr(
          "available_memory_proportion_bwd"),
      options=op.get_attr("options_bwd"),
  )
  return [g[0], g[1], g[2], g[3], g[4], None]


@ops.RegisterGradient("PopnnGRULayer")
def _popnn_gru_layer_backward(op, *grads):
  """Gradients for the PopnnGRULayer op."""
  if not op.get_attr("is_training"):
    raise ValueError(
        "To use PopnnGRULayer in gradients, is_training must be set to True.")
  return gen_popnn_ops.popnn_gru_layer_backprop(
      inputs=op.inputs[0],
      initial_state=op.inputs[1],
      kernel=op.inputs[2],
      biases=op.inputs[3],
      output=op.outputs[0],
      output_state=op.outputs[1],
      intermediates=op.outputs[2],
      output_backprop=grads[0],
      output_state_backprop=grads[1],
      activation=op.get_attr("activation"),
      recurrent_activation=op.get_attr("recurrent_activation"),
      num_channels=op.get_attr("num_channels"),
      partials_dtype=op.get_attr("partials_dtype"),
      is_training=op.get_attr("is_training"),
      reset_after=op.get_attr("reset_after"),
      available_memory_proportion_bwd=op.get_attr(
          "available_memory_proportion_bwd"),
      options=op.get_attr("options_bwd"),
  )


@ops.RegisterGradient("PopnnDynamicGRULayer")
def _popnn_dynamic_gru_layer_backward(op, *grads):
  """Gradients for the PopnnDynamicGRULayer op."""
  if not op.get_attr("is_training"):
    raise ValueError(
        "To use PopnnDynamicGRULayer in gradients, is_training must be "
        "set to True.")
  g = gen_popnn_ops.popnn_dynamic_gru_layer_backprop(
      inputs=op.inputs[0],
      initial_state=op.inputs[1],
      kernel=op.inputs[2],
      biases=op.inputs[3],
      seq_len=op.inputs[4],
      output=op.outputs[0],
      output_state=op.outputs[1],
      intermediates=op.outputs[2],
      output_backprop=grads[0],
      output_state_backprop=grads[1],
      activation=op.get_attr("activation"),
      recurrent_activation=op.get_attr("recurrent_activation"),
      num_channels=op.get_attr("num_channels"),
      partials_dtype=op.get_attr("partials_dtype"),
      is_training=op.get_attr("is_training"),
      reset_after=op.get_attr("reset_after"),
      available_memory_proportion_bwd=op.get_attr(
          "available_memory_proportion_bwd"),
      options=op.get_attr("options_bwd"),
  )
  return [g[0], g[1], g[2], g[3], None]


@ops.RegisterGradient("PopnnAUGRULayer")
def _popnn_augru_layer_backward(op, *grads):
  """Gradients for the PopnnAUGRULayer op."""
  if not op.get_attr("is_training"):
    raise ValueError(
        "To use PopnnAUGRULayer in gradients, is_training must be set to True."
    )
  g = gen_popnn_ops.popnn_augru_layer_backprop(
      inputs=op.inputs[0],
      initial_state=op.inputs[1],
      kernel=op.inputs[2],
      biases=op.inputs[3],
      att_score=op.inputs[5],
      seq_len=op.inputs[4],
      output=op.outputs[0],
      output_state=op.outputs[1],
      intermediates=op.outputs[2],
      output_backprop=grads[0],
      output_state_backprop=grads[1],
      activation=op.get_attr("activation"),
      recurrent_activation=op.get_attr("recurrent_activation"),
      num_channels=op.get_attr("num_channels"),
      partials_dtype=op.get_attr("partials_dtype"),
      is_training=op.get_attr("is_training"),
      reset_after=op.get_attr("reset_after"),
      available_memory_proportion_bwd=op.get_attr(
          "available_memory_proportion_bwd"),
      options=op.get_attr("options_bwd"),
  )
  return [g[0], g[1], g[2], g[3], None, g[4]]
