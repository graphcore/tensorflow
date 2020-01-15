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
# =============================================================================
"""
Keras' Sequential class for the IPU.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow import keras
from tensorflow.python.framework import ops


class Sequential(keras.Sequential):
  """Implementation of tf.keras.Sequential which explicitly targets the IPU.
  """
  def __init__(
      self,
      layers=None,
      name=None,
      ipu_device="/device:IPU:0",
  ):
    """Create a Sequential object.

    layers and name:
      See tensorflow.keras.Sequential documentation.

    ipu_device:
      Target device to explicitly target when adding layers to the sequence.
    """
    super().__init__(layers=layers, name=name)
    self._device = ipu_device

  def add(self, layer):
    with ops.device(self._device):
      super().add(layer)

  def call(self, inputs, training=None, mask=None):  # pylint: disable=redefined-outer-name
    with ops.device(self._device):
      return super().call(inputs, training, mask)

  def build(self, input_shape=None):  # pylint: disable=useless-super-delegation
    # Note: This override is needed to preserve the behaviour of the weights
    # property of the original Sequential class.
    super().build(input_shape)
