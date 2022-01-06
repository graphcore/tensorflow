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
from tensorflow.compiler.plugin.poplar.ops import gen_poprand_ops
"""
    These gradient function should *never* be called directly.
"""
@ops.RegisterGradient("IpuDropout")
def _dropout_grad(op, *grads):
  """Gradients for the IPU dropout op."""
  seed = op.outputs[1]
  rate = op.get_attr("rate")
  scale = op.get_attr("scale")
  noise_shape = op.get_attr("noise_shape")

  return [
      gen_poprand_ops.ipu_dropout_with_seed_and_reference(
          grads[0],
          seed=seed,
          reference=op.outputs[2],
          rate=rate,
          scale=scale,
          noise_shape=noise_shape)[0]
  ]


@ops.RegisterGradient("IpuDropoutWithSeed")
def _dropout_grad_with_seed(op, *grads):
  # The seed has no gradient.
  return _dropout_grad(op, *grads) + [None]
