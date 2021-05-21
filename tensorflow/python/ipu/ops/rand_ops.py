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
Poprand operators
~~~~~~~~~~~~~~~~~
"""
from tensorflow.compiler.plugin.poplar.ops import gen_poprand_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


def dropout(x, rate=0.5, noise_shape=None, seed=None, name=None, **kwargs):
  """This targets the PopLibs Poprand operation, optimized for execution
  on the IPU.

  With probability `rate`, drops elements of `x`. Inputs which are kept are
  scaled up by `1 / (1 - rate)` such that the expected sum is unchanged.

  Args:
    x: The input tensor.
    rate: The probability that a given element will be zeroed out.
    noise_shape: An optional parameter that determines the shape of the dropout.
                 Regular, unshaped dropout used if not specified.
    seed: An optional two-element tensor-like object (`tf.Tensor`, a numpy
      array or Python list/tuple) containing a pair of 32-bit integers that will
      be used to seed the random number generator that generates the dropout
      mask.
    name: Optional op name.

  Returns:
    A tensor which has some nodes set to zero, as randomly selected based on
    other parameters.
  """

  # Rate is a probability between 0 and 1. Specifically the rate that a variable
  # will be dropped out.
  if rate >= 1.0 or rate < 0.0:
    raise ValueError("The rate must be in the range [0, 1), but was %s" % rate)

  if noise_shape:
    x_shape = x.get_shape().as_list()
    if len(x_shape) != len(noise_shape):
      raise ValueError("The length of noise_shape must equal the rank of x.")

    for i, j in zip(x_shape, noise_shape):
      if j == 1:
        continue

      if i != j:
        raise ValueError("Dimension mismatch, %d != %d." % (i, j))

  # The ipu_dropout op uses the old tensorflow method where the rate is the
  # probability of keeping rather than dropping.
  keep_prob = 1 - rate
  scale = 1 / keep_prob
  ref = kwargs.pop("ref", True)

  if seed is None and ref:
    return gen_poprand_ops.ipu_dropout(x,
                                       rate=keep_prob,
                                       scale=scale,
                                       noise_shape=noise_shape,
                                       name=name)[0]

  if seed is None:
    return gen_poprand_ops.ipu_dropout_no_ref(x,
                                              rate=keep_prob,
                                              scale=scale,
                                              noise_shape=noise_shape,
                                              name=name)[0]
  seed = ops.convert_to_tensor(seed)
  if seed.shape != [2]:
    raise ValueError("Expected the seed to have a shape [2], but got %s." %
                     (str(seed.shape)))

  if ref:
    return gen_poprand_ops.ipu_dropout_with_seed(x,
                                                 seed=seed,
                                                 rate=keep_prob,
                                                 scale=scale,
                                                 noise_shape=noise_shape,
                                                 name=name)[0]

  return gen_poprand_ops.ipu_dropout_with_seed_no_ref(x,
                                                      seed=seed,
                                                      rate=keep_prob,
                                                      scale=scale,
                                                      noise_shape=noise_shape,
                                                      name=name)[0]
