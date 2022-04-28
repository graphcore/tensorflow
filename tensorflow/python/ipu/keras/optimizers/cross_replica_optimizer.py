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
Optimizer wrapper for replicated graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.ipu.optimizers.cross_replica_optimizer import apply_cross_replica_op_single
from tensorflow.python.ipu.keras.optimizers import IpuOptimizer
from tensorflow.python.util import deprecation


class CrossReplicaOptimizer(IpuOptimizer):
  """An optimizer that averages gradients across IPU replicas."""
  @deprecation.deprecated(None, "This optimizer will be removed.")
  def __init__(self, opt, name="CrossReplicaOptimizer"):
    """Construct a new cross-replica optimizer.

    Args:
      opt: An existing `Optimizer` to encapsulate.
      kwargs: Keyword arguments to pass onto the wrapper optimizer
    """

    super().__init__(opt, name=name)

  def _resource_apply_dense(self, grad, handle, apply_state):
    """Apply gradient to variable referenced by `handle`.

    Sums the gradient contributions across replicas,
    and then applies the wrapped optimizer.

    Args:
      grad: The gradient to be applied.
      handle: A handle to the variable to apply the gradient to.
      apply_state: State needed by wrapped optimizer to apply gradients.

    Returns:
      The updated variable.
    """
    (new_grad, new_var) = apply_cross_replica_op_single(grad, handle)
    return super()._resource_apply_dense(  # pylint: disable=protected-access
        new_grad,
        new_var,
        apply_state=apply_state)

  @classmethod
  def from_config(cls, config, custom_objects=None):
    """Creates a `CrossReplicaOptimizer` from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same optimizer from the config
    dictionary.

    Arguments:
        config: A Python dictionary, typically the output of get_config.
        custom_objects: A Python dictionary mapping names to additional Python
          objects used to create this optimizer, such as a function used for a
          hyperparameter.

    Returns:
        A `CrossReplicaOptimizer` instance.
    """
    config = config.copy()
    IpuOptimizer._verify_config(config)
    inner_config = config.pop('inner_optimizer_config')
    inner_type = config.pop('inner_optimizer_type')
    inner_opt = inner_type(**inner_config)

    return CrossReplicaOptimizer(inner_opt, **config)
