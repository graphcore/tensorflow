# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""
Gradient utility operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_grad_util_ops
from tensorflow.python.ipu.eager import backprop as ipu_backprop
from tensorflow.python.util import nest


def capture_upstream_gradients(args, tag=""):
  """A function that wraps an operation for the purpose of
  capturing and storing upstream gradients in the model.
  Performs a no-op on a forward pass and is equivelant to the Identity
  function in value. However, the incoming gradients of the wrapped
  operation will be made available via the `captured_gradients`
  attribute of `tensorflow.python.ipu.eager.backprop.GradientCaptureContext`
  and `tensorflow.python.ipu.eager.backprop.GradientCaptureTape`.

  Example

  .. code-block:: python
    opt = SGD()
    x = Variable(3.0)
    with GradientCaptureContext() as gcc:
      o = x**2
      p = capture_upstream_gradients(o, tag="tanh_grad")
      y = math_ops.tanh(p)
      grads_wrt_vars = opt.get_gradients(y, x)

    captured_grads = gcc.captured_gradients
    tanh_grad = grads_wrt_vars["tanh_grad"]

  Args:
    args: The return value(s) of the wrapped operation.
    tag: The name tag used to retrieve the captured gradients.

  Returns:
    `args`

  Raises:
    RuntimeError: If used outside of a
    `tensorflow.python.ipu.eager.backprop.GradientCaptureContext` or
    `tensorflow.python.ipu.eager.backprop.GradientCaptureTape` context
    if its gradient is taken.
  """
  out = gen_grad_util_ops.capture_upstream_gradients_fwd_no_op(
      nest.flatten(args), tag=tag)
  return nest.pack_sequence_as(args, out)
