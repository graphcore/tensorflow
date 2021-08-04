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
    Construct a GradientAccumulationOptimizer.

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

    if num_mini_batches < 1:
      raise ValueError("num mini batches must be >= 1")

    IpuOptimizer.__init__(self, opt, name=name)
    self._num_mini_batches = num_mini_batches
    self._offload_weight_update_variables = self.bool_to_three_state(
        offload_weight_update_variables,
        threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_UNDEFINED))
    self._replicated_optimizer_state_sharding = self.bool_to_three_state(
        replicated_optimizer_state_sharding,
        self._offload_weight_update_variables)
    self._dtype = dtype

  def _resource_apply_dense(self, grad, handle, apply_state):
    """Apply gradient to variable referenced by `handle`.

    Args:
      grad: The gradient to be applied.
      handle: A handle to the variable to apply the gradient to.
      apply_state: State passed down to wrapped optimzier's apply functions.
    Returns:
      The updated variable.
    """
    (acc_grad,
     acc_var) = GradientAccumulationOptimizerV2.create_accumulated_grads(
         grad, handle, self._dtype, self._num_mini_batches)

    # Create an explicit function call for the apply gradients - note that we
    # allow external captures here.
    apply_grad_ops = []

    def resource_update_():
      updated_var = self._opt._resource_apply_dense(  # pylint: disable=protected-access
          acc_grad, acc_var, apply_state)
      if updated_var is not None:
        apply_grad_ops.append(updated_var)

    return GradientAccumulationOptimizerV2.apply_gradient_accumulation(
        resource_update_,
        self._opt._name,  # pylint: disable=protected-access
        apply_grad_ops,
        self._offload_weight_update_variables,
        self._replicated_optimizer_state_sharding,
        self._num_mini_batches)
