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
Base IPU Keras layer
~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.util import deprecation


class IPULayer(Layer):
  @deprecation.deprecated(
      None,
      "The IPULayer keras layer has been moved to IPU TensorFlow Addons and "
      "will be removed from TensorFlow in a future release.")
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def _check_unsupported(self, arg, arg_name, method="__init__"):
    if arg:
      raise NotImplementedError(
          "ipu.keras.%s does not support %s"
          " argument %s. It is included for API consistency"
          "with keras.%s." %
          (self.__class__.__name__, method, arg_name, self.__class__.__name__))
