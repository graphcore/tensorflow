# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
Control flow operations.
~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops

from tensorflow.python.util import nest


def barrier(tensors, insert_barrier_for_gradients=False, name=None):
  """
  A control flow operation to force the scheduling of operations in the Poplar
  XLA backend.

  For example given the following program:

  .. code-block:: python

    def func(a, b, c, d):
      e = a + b
      f = c + d
      g = e + a
      return f, g

    The operations `f` and `g` are independent of each other meaning that either
    `f` or `g` can execute first. However if we want to force `f` to execute
    first, we can insert a barrier operation:

  .. code-block:: python

    def func(a, b, c, d):
      e = a + b
      f = c + d
      f, a = ipu.control_flow_ops.barrier([f, a])
      g = e + a
      return f, g

  This will result in `f` executing before `g` as now there is a data dependency
  between the operations.

  Args:
    tensors: A tensor or a structure of tensors which all have to be executed
      before the outputs of the barrier operation can be used.

  Returns:
    A tensor or a structure of tensors which matches shape and type of the
    `tensors` arg.
  """
  flat_tensors = nest.flatten(tensors)
  barrier_tensors = gen_poputil_ops.ipu_barrier(
      flat_tensors,
      insert_barrier_for_gradients=insert_barrier_for_gradients,
      name=name)
  return nest.pack_sequence_as(tensors, barrier_tensors)
