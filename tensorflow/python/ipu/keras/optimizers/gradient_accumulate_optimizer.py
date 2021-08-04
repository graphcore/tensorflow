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
# =============================================================================
"""
Optimizer wrapper to accumulate gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.ipu.keras.optimizers import IpuOptimizer
from tensorflow.compiler.plugin.poplar.driver import threestate_pb2
from tensorflow.python.ipu.optimizers import GradientAccumulationOptimizerV2


class GradientAccumulationOptimizer(IpuOptimizer):
  """An optimizer which performs the weight update after multiple batches
  have been accumulated.
  """
  @staticmethod
  def bool_to_three_state(value, default):
    if value is None:
      return default
    elif value:
      return threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_ON)
    return threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_OFF)

  def __new__(cls, opt, num_mini_batches, *nargs, **kwargs):  #pylint: disable=unused-argument
    if num_mini_batches == 1:
      return opt
    return super(GradientAccumulationOptimizer, cls).__new__(cls)

  def __init__(self,
               opt,
               num_mini_batches,
               offload_weight_update_variables=None,
               replicated_optimizer_state_sharding=False,
               dtype=None,
               name="GradientAccumulationOptimizer"):
    """
    Construct a GradientAccumulationOptimizer. Note this doesn't divide
    by the number of mini-batches.

    Args:
      opt: An existing optimizer to encapsulate.
      num_mini_batches: The number of mini-batches the gradients
                        will be accumulated for.
      offload_weight_update_variables: When enabled, any `tf.Variable` which is
        only used by the weight update of the pipeline (for example the
        accumulator variable when using the `tf.MomentumOptimizer`), will be
        stored in the remote memory. During the weight update this variable will
        be streamed onto the device and then streamed back to the remote memory
        after it has been updated. Requires the machine to be configured with
        support for `Poplar remote buffers`. Offloading variables into remote
        memory can reduce maximum memory liveness, but can also increase the
        computation time of the weight update.
        When set to `None` the variables will be placed in either in-processor
        or remote memory automatically based on the current best placement
        strategy.
      replicated_optimizer_state_sharding: If True, any any `tf.Variable` which
        is offloaded will be partitioned across the replicas. A collective
        all-gather will be inserted to restore the tensor on each replica.
        If `None`, this value will match the value of
        `offload_weight_update_variables`.
      dtype: The data type used for the gradient accumulation buffer.
        One of:
          - `None`: Use an accumulator of the same type as the variable type.
          - A `DType`: Use this type for all the accumulators.
          - A callable that takes the variable and returns a `DType`: Allows
            specifying the accumulator type on a per-variable basis.

        The gradients passed to `Optimizer.apply_gradients` will have the dtype
        requested here. If that dtype is different from the variable dtype
        a cast is needed at some point to make them compatible. If you want
        to cast the gradients immediately, you can wrap your optimizer in the
        `MapGradientOptimizer` with a `tf.cast`.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientAccumulationOptimizer".
    """
    raise NotImplementedError("""Cannot use v2 optimizer. Have to use
                GradientAccumulationOptimizerV2 instead (found in 
                tensorflow.python.ipu.optimizers)""")
