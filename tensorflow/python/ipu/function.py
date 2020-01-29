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
# =============================================================================
"""
Graph functions for the IPU.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.eager import function as function_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_decorator


# This class ensures that when the function passed is called, it is done with
# an IPU device context.
class IpuFunction(def_function.Function):
  def __init__(self,
               python_function,
               name,
               input_signature=None,
               autograph=True,
               experimental_implements=None,
               experimental_autograph_options=None,
               experimental_relax_shapes=False,
               ipu_device="/device:IPU:0"):
    super().__init__(
        python_function,
        name,
        input_signature=input_signature,
        autograph=autograph,
        experimental_implements=experimental_implements,
        experimental_autograph_options=experimental_autograph_options,
        experimental_relax_shapes=experimental_relax_shapes,
        experimental_compile=True)
    self._device = ipu_device

  def _call(self, *args, **kwds):
    with ops.device(self._device):
      return super()._call(*args, **kwds)


def function(func=None,
             input_signature=None,
             autograph=True,
             ipu_device="/device:IPU:0",
             experimental_implements=None,
             experimental_autograph_options=None,
             experimental_relax_shapes=False):
  """Compiles a function into a callable TensorFlow graph.

  The function will be compiled using XLA and the Poplar back end. Both
  a python module function, and a member of a class can be annotated.

  This function annotation is used in a similar manner to tf.function,
  but it explicitly targets the IPU.

  See the documentation for tf.function for examples and discussions
  of proper use.
  """

  if input_signature is not None:
    function_lib.validate_signature(input_signature)

  def decorated(inner_function):
    try:
      name = inner_function.__name__
    except AttributeError:
      name = "function"
    return tf_decorator.make_decorator(
        inner_function,
        IpuFunction(
            inner_function,
            name,
            input_signature=input_signature,
            autograph=autograph,
            experimental_autograph_options=experimental_autograph_options,
            experimental_implements=experimental_implements,
            experimental_relax_shapes=experimental_relax_shapes,
            ipu_device=ipu_device))

  if func is not None:
    return decorated(func)

  return decorated
