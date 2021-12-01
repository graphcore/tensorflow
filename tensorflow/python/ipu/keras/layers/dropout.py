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

from tensorflow.python.ipu.ops import rand_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util import deprecation


class Dropout(Layer):
  """Dropout layer optimized for running on the IPU.

  The Dropout layer randomly sets input units to 0 with a frequency of `rate`
  at each step during training. Inputs not set to 0 are scaled up by
  `1/(1 - rate)` such that the expected sum is unchanged.

  Note that the Dropout layer only applies when `training` is set to True, so
  no values are dropped during inference.

  Args:
    rate: Float between 0 and 1. Fraction of the input units to drop.
    noise_shape: 1D integer tensor representing the shape of the binary
      dropout mask that will be multiplied with the input.
    seed: An optional two-element tensor-like object (`tf.Tensor`, a numpy
      array or Python list/tuple) containing a pair of 32-bit integers that will
      be used to seed the random number generator that generates the dropout
      mask.
  """
  @deprecation.deprecated(
      None,
      "The Dropout keras layer has been moved to IPU TensorFlow Addons and "
      "will be removed from TensorFlow in a future release.")
  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    self.seed = seed
    self.rate = rate
    self.noise_shape = noise_shape
    self.ref = kwargs.pop("ref", True)
    super(Dropout, self).__init__(**kwargs)

  # pylint: disable=useless-super-delegation
  def build(self, input_shape):
    super(Dropout, self).build(input_shape)

  # pylint: disable=arguments-differ
  def call(self, inputs, training=None):
    """ Perform dropout.

    Args:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave
        in training mode (adding dropout) or in inference mode (doing
        nothing).

    Returns:
      In training mode, a tensor which has some nodes set to zero, as randomly
      selected based on other parameters. In inference mode, a tensor that is
      identical to the input tensor.
    """
    def dropped_inputs():
      return rand_ops.dropout(inputs,
                              seed=self.seed,
                              rate=self.rate,
                              noise_shape=self.noise_shape,
                              ref=self.ref,
                              name=self.name)

    output = K.in_train_phase(dropped_inputs,
                              lambda: array_ops.identity(inputs),
                              training=training)

    return output

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    return {
        'rate': self.rate,
        'noise_shape': self.noise_shape,
        'seed': self.seed,
        'ref': self.ref
    }
