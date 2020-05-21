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
"""
Popnn random operators
~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.ops import gen_poprand_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


def dropout(x, seed=None, rate=0.5, scale=1, seed_modifier=1, name=None):
  """This targets the poplibs popnn dropout operation, optimized for execution
  on the IPU.

  Args:
    x: The input tensor.
    seed: An optional two-element tensor-like object (`tf.Tensor`, a numpy array
      or Python list/tuple), representing the random seed that will be used to
      create the distribution for dropout.
    rate: The probability that a given element will be zeroed out.
    scale: An optional factor to apply to all other elements.
    seed_modifier: An optional parameter given to poplar which uses it to modify
                   the seed.
    name: Optional op name.

  Returns:
    A `Tensor` which has some nodes set to zero, as randomly selected based on
    other parameters.
  """

  # Rate is a probability between 0 and 1. Specifically the rate that a variable
  # will be dropped out.
  if rate > 1.0 or rate < 0.0:
    raise ValueError("Rate must be between 0.0 and 1.0" % rate)

  is_using_user_seed = True
  if seed is None:
    is_using_user_seed = False
    # Create empty placeholder we will generate a random one internally.
    seed = array_ops.zeros([2], dtypes.int32)

  seed = ops.convert_to_tensor(seed)
  if seed.shape != [2]:
    raise ValueError("Expected the seed to have a shape [2], but got %s." %
                     (str(seed.shape)))

  # We transfrom rate to be the change an individual node will dropout as
  # ipu_dropout is using the old tensorflow method that rate is the probability
  # that value is kept rather than disgarded.
  return gen_poprand_ops.ipu_dropout(x,
                                     seed=seed,
                                     rate=(1 - rate),
                                     scale=scale,
                                     name=name,
                                     is_using_user_seed=is_using_user_seed,
                                     seed_modifier=seed_modifier)[0]
