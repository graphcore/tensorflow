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

from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops


# This class ensures that when the function passed is called, it is done with
# an IPU device context.
class IpuFunction():
  def __init__(self, fn, device):
    self._call = fn
    self._device = device

  def __call__(self, *args, **kwds):
    with ops.device(self._device):
      return self._call(*args, **kwds)


def function(func=None,
             input_signature=None,
             autograph=True,
             ipu_device="/device:IPU:0",
             experimental_implements=None,
             experimental_autograph_options=None):
  """Compiles a function into a callable TensorFlow graph.

      The function will be compiled using XLA and the Poplar back end.

      This function annotation is used in a similar manner to tf.function,
      but it explicitly targets the IPU.
      """
  f = def_function.function(
      func,
      input_signature=input_signature,
      autograph=autograph,
      experimental_implements=experimental_implements,
      experimental_autograph_options=experimental_autograph_options,
      experimental_relax_shapes=False,
      experimental_compile=True)

  decorator = IpuFunction(f, ipu_device)

  # We wrap this so we are still returning a function rather than an instance
  # of IpuFunction, this is so that if we are annotating a member function then
  # a call to foo.wrapper(args, kwargs) will add "self=foo" to the argument list.
  def wrapper(*args, **kwargs):
    return decorator(*args, **kwargs)

  return wrapper
