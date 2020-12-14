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
@ops.RegisterGradient("PopnnGroupNormTraining")
def _popnn_group_norm_backward(op, *grads):
  """Gradients for the PopnnGroupNormTraining op."""
  return gen_popnn_ops.popnn_group_norm_grad(
      inputs=op.inputs[0],
      gamma=op.inputs[1],
      mean=op.outputs[1],
      inv_std_dev=op.outputs[2],
      output_backprop=grads[0],
      data_format=op.get_attr("data_format"),
      epsilon=op.get_attr("epsilon"),
      num_groups=op.get_attr("num_groups"),
      strided_channel_grouping=op.get_attr("strided_channel_grouping"))
