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
Keras Model interfaces for IPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


class _IpuModelBase:
  """Base class for IPU Keras models"""
  def __init__(self, *args, **kwargs):
    raise RuntimeError("_IpuModelBase has been deleted. Use tf.keras instead.")


class IPUSequential:
  def __init__(self, *args, **kwargs):
    raise RuntimeError(
        "IPUSequential has been deleted. Use tf.keras.Sequential within a "
        "`tensorflow.python.ipu.ipu_strategy.IPUStrategy` instead.")


class IPUModel:
  def __init__(self, *args, **kwargs):
    raise RuntimeError(
        "IPUModel has been deleted. Use tf.keras.Model within a "
        "`tensorflow.python.ipu.ipu_strategy.IPUStrategy` instead.")


Model = IPUModel
Sequential = IPUSequential
