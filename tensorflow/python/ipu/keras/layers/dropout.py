# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
Dropout Keras layer
~~~~~~~~~~~~~~~~~~~
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.plugin.poplar.ops import gen_poprand_ops
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.util.tf_export import keras_export


@keras_export(v1=['keras.ipu.layers.Dropout'])
class Dropout(Layer):
  """Base class for implementing XLA and Popnn compatible Dropout layer.
    """
  def __init__(self,
               seed=None,
               rate=0.5,
               scale=1,
               seed_modifier=1,
               name=None,
               dtype=None):
    """Creates a Dropout model.

        Args:
            seed: A Python integer to use as random seed.
            rate: Float between 0 and 1. Fraction of the input units to drop.
            scale: An optional factor to apply to all other elements.
            seed_modifier: An optional parameter given to poplar which
                         uses it to modify the seed.
            name: Optional op name.
            dtype: tf.float16 or tf.float32

        Returns:
            A `Tensor` which has some nodes set to zero, as randomly
            selected based on other parameters.
        """
    super(Dropout, self).__init__(dtype=dtype, name=name)
    self.built = False
    self.seed = seed
    self.rate = rate
    self.scale = scale
    self.seed_modifier = seed_modifier

    if seed is None:
      # User did not provide a seed
      self.seed = [0, 0]
      self.is_using_user_seed = False
    else:
      # User provided a seed
      self.is_using_user_seed = True

  def saveable(self):
    raise NotImplementedError(
        "This cell does not yet support object-based saving. File a feature "
        "request if this limitation bothers you.")

  # pylint: disable=useless-super-delegation
  def build(self, input_shape):
    super(Dropout, self).build(input_shape)

  # pylint: disable=arguments-differ
  def call(self, x):
    return gen_poprand_ops.ipu_dropout(
        x,
        seed=self.seed,
        user_seed=1,
        rate=(1.0 - self.rate),
        scale=self.scale,
        name=self.name,
        is_using_user_seed=self.is_using_user_seed,
        seed_modifier=self.seed_modifier)
