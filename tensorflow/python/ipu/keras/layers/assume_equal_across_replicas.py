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
Assume Equal Across Replicas IPU Keras layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ipu.ops import cross_replica_ops


class AssumeEqualAcrossReplicas(Layer):
  """
  Layer for marking values as equal across replicas to try and prevent divergent
  control flow compilation errors.

  Divergent control flow describes the situation where program flow differs
  among replicas. This happens when the value of a conditional is not the same
  across all replicas. This is a problem if the conditional body requires a
  cross-replica sync, as only some replicas will reach it. If this happens,
  the execution will hang as the operation waits for all replicas to sync.

  To warn the user about this, Poplar checks for divergent control flow during
  compilation. However since the values of tensors are unknown at compilation
  time it can't be certain whether a tensor will lead to divergent control
  flow or not. `assume_equal_across_replicas` can be used to mark tensors
  which are equal across all replicas and in doing so prevents them causing
  divergency errors, if used in a conditional.

  Args:
    inplace: A bool for controlling whether or not the given tensor(s) is copied
      or operated on inplace. This is needed when using
      `AssumeEqualAcrossReplicas` with tensor slices.
  """
  def __init__(self, inplace=False, **kwargs):
    super().__init__(**kwargs)
    self._inplace = inplace

  def call(self, inputs, **kwargs):
    return cross_replica_ops.assume_equal_across_replicas(
        inputs, self._inplace)
