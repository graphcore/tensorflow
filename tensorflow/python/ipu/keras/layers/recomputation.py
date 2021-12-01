# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
Recomputation IPU Keras layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.util import deprecation


class RecomputationCheckpoint(Layer):
  """
  Layer for checkpointing values in a computational pipeline stage.
  When recomputation is enabled, these values will not be recomputed and they
  will be stored in memory instead.

  This layer can reduce memory liveness peaks when using recomputation if
  there are too many activations which need to be recomputed before the
  backpropagation operations can be executed.

  This layer should be used with the
  `RecomputationMode.RecomputeAndBackpropagateInterleaved` pipelining
  recomputation mode.

  Note that this layer has no effect when used with the
  `RecomputationMode.RecomputeThenBackpropagate` pipelining
  recomputation mode.
  """
  @deprecation.deprecated(
      None, "The RecomputationCheckpoint keras layer has been moved to IPU "
      "TensorFlow Addons and will be removed from TensorFlow in a future "
      "release.")
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def call(self, inputs, **kwargs):
    """
    Checkpoint the input tensors.

    Args:
      inputs: A tensor or a structure of tensors which should be checkpointed.

    Returns:
      A tensor or a structure of tensors which matches shape and type of
      `inputs`.
    """
    return pipelining_ops.recomputation_checkpoint(inputs, name=self.name)

  def get_config(self):
    return {}
