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

from tensorflow.python.ipu import rand_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops


class Dropout(Layer):
  """Base class for implementing XLA and Popnn compatible Dropout layer.
  """
  def __init__(self,
               rate=0.5,
               noise_shape=None,
               seed=None,
               scale=1,
               seed_modifier=1,
               **kwargs):
    """Creates a Dropout model.

    Args:
      rate: Float between 0 and 1. Fraction of the input units to drop.
      seed: An optional two-element tensor-like object (`tf.Tensor`, a numpy
        array or Python list/tuple), representing the random seed that will be
        used to create the distribution for dropout.
      noise_shape: 1D integer tensor representing the shape of the binary
        dropout mask that will be multiplied with the input.
      scale: An optional factor to apply to all retained elements.
      seed_modifier: An optional parameter given to poplar which uses it to
        modify the seed.

    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave
        in training mode (adding dropout) or in inference mode (doing
        nothing).

    Returns:
      A `Tensor` which has some nodes set to zero, as randomly selected based on
      other parameters.
    """
    super(Dropout, self).__init__(**kwargs)
    self.built = False
    self.seed = seed
    self.rate = rate
    self.scale = scale
    self.seed_modifier = seed_modifier

    if noise_shape is not None:
      raise ValueError(
          "ipu.keras.Dropout does not support `noise_shape` parameter")

    if seed is None:
      # User did not provide a seed
      self.seed = [0, 0]
      self.is_using_user_seed = False
    else:
      # User provided a seed
      self.is_using_user_seed = True

  # pylint: disable=useless-super-delegation
  def build(self, input_shape):
    super(Dropout, self).build(input_shape)

  # pylint: disable=arguments-differ
  def call(self, x, training=True):
    if not isinstance(training, bool):
      raise ValueError(
          "ipu.keras.Dropout does not support a dynamic training parameter.  "
          "Pass a boolean True or False.  If you are using keras Sequential, "
          "then use a different model class.")

    if training:
      output = rand_ops.dropout(x,
                                seed=self.seed,
                                rate=self.rate,
                                scale=self.scale,
                                seed_modifier=self.seed_modifier,
                                name=self.name)
    else:
      output = array_ops.identity(x)

    return output

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'rate': self.rate,
        'noise_shape': self.noise_shape,
        'seed': self.seed,
        'scale': self.scale,
        'seed_modifier': self.seed_modifier
    }
    base_config = super(Dropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
