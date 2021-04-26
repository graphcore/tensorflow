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
Optimizer wrapper for sharded graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.framework import ops
from tensorflow.python.ipu import sharding
from tensorflow.python.training import optimizer
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2


class ShardedOptimizer(optimizer.Optimizer):
  def __init__(self, optimizer):
    """Construct a new sharded optimizer.

    Args:
      optimizer: The optimizer to wrap.
    """
    if isinstance(optimizer, OptimizerV2):
      raise ValueError("Should use optimizers in ipu.keras.optimizers "
                       "to wrap V2 optimizers")

    super(ShardedOptimizer, self).__init__(False, name="ShardedOptimizer")
    self._optimizer = optimizer

  def compute_gradients(self, loss, var_list=None, **kwargs):
    kwargs['colocate_gradients_with_ops'] = True
    ret = self._optimizer.compute_gradients(loss, var_list=var_list, **kwargs)
    sharding.propagate_sharding(ops.get_default_graph())
    return ret

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    ret = self._optimizer.apply_gradients(grads_and_vars, global_step, name)
    sharding.propagate_sharding(ops.get_default_graph())
    return ret

  def get_slot_names(self, *args, **kwargs):
    return self._optimizer.get_slot_names(*args, **kwargs)

  def variables(self):
    return self._optimizer.variables()
