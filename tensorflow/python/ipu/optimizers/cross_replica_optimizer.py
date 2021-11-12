# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.framework import ops
from tensorflow.python.ipu.ops import cross_replica_ops
from tensorflow.python.ipu.optimizers import IpuOptimizer
from tensorflow.python.training import optimizer


def apply_cross_replica_op_single(grad, var):
  if grad is None:
    return (grad, var)
  with ops.colocate_with(grad):
    return (cross_replica_ops.cross_replica_mean(grad), var)


def apply_cross_replica_op(grads_and_vars):
  summed_grads_and_vars = []
  for (grad, var) in grads_and_vars:
    summed_grads_and_vars.append(apply_cross_replica_op_single(grad, var))
  return summed_grads_and_vars


class CrossReplicaOptimizer(IpuOptimizer):
  """An optimizer that averages gradients across IPU replicas."""
  def __init__(self, opt, name="CrossReplicaOptimizer"):
    """Construct a new cross-replica optimizer.

    Args:
      opt: An existing `Optimizer` to encapsulate.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "CrossReplicaOptimizer".
    """
    super(CrossReplicaOptimizer, self).__init__(opt, name=name)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    Calls popops_cross_replica_sum.cross_replica_sum() to sum gradient
    contributions across replicas, and then applies the real optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      An `Operation` that applies the gradients. If `global_step` was not None,
      that operation also increments `global_step`.

    Raises:
      ValueError: If the grads_and_vars is malformed.
    """
    summed_grads_and_vars = apply_cross_replica_op(grads_and_vars)
    return self._opt.apply_gradients(summed_grads_and_vars, global_step, name)
