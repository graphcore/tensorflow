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
"""Gradients for Popfloat operators."""

from tensorflow.python.framework import ops
"""
    These gradient function should *never* be called directly.
"""
@ops.RegisterGradient("CalcGfloatParams")
def _calc_gfloat_params_backward(op, *grads):
  """Gradients for the CalcGfloatParams op."""
  return None


@ops.RegisterGradient("CastNativeToGfloat")
def _cast_native_to_gfloat_backward(op, *grads):
  """Gradients for the CastToGfloat op."""
  return [grads[0], None]


@ops.RegisterGradient("CastGfloatToNative")
def _cast_gfloat_to_native_backward(op, *grads):
  """Gradients for the CastFromGfloat op."""
  return [grads[0], None]
