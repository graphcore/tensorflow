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
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2


# Optimizer wrapper that captures intermedite kernel/weight values into the given
# outfeed. This lets us compare the value of weights across replicas.
class KernelLoggingOptimizer(Optimizer):
  def __init__(self, outfeed_queue, wrapped_optimizer, model=None):
    super(KernelLoggingOptimizer, self).__init__(False,
                                                 "KernelLoggingOptimizer")

    self._wrapped_optimizer = wrapped_optimizer
    self._outfeed_queue = outfeed_queue
    self._model = model

    self._using_v2_optimizer = isinstance(self._wrapped_optimizer,
                                          optimizer_v2.OptimizerV2)

  def compute_gradients(self, loss, var_list=None, **kwargs):  # pylint: disable=arguments-differ,unused-argument
    if isinstance(self._wrapped_optimizer, optimizer_v2.OptimizerV2):
      grads = self._wrapped_optimizer.get_gradients(
          loss, self._model.trainable_weights)
      grads_and_vars = list(zip(grads, self._model.trainable_weights))
    else:
      grads_and_vars = self._wrapped_optimizer.compute_gradients(
          loss, var_list=var_list, **kwargs)

    return grads_and_vars

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    kernels = []
    for _, var in list(reversed(grads_and_vars)):
      if "kernel" in var.name:
        kernels.append(var)
    outfeed = self._outfeed_queue.enqueue(kernels)

    with ops.control_dependencies([outfeed]):
      if self._using_v2_optimizer:
        return self._wrapped_optimizer.apply_gradients(grads_and_vars)

      return self._wrapped_optimizer.apply_gradients(grads_and_vars,
                                                     global_step, name)

  def _apply_dense(self, grad, var):
    return self._wrapped_optimizer._apply_dense(grad, var)  # pylint: disable=protected-access

  def _resource_apply_dense(self, grad, handle):
    return self._wrapped_optimizer._resource_apply_dense(grad, handle)  # pylint: disable=protected-access

  def _apply_sparse(self, grad, var):
    return self._wrapped_optimizer._apply_sparse(grad, var)  # pylint: disable=protected-access

  def _resource_apply_sparse(self, grad, handle, indices):
    return self._wrapped_optimizer._resource_apply_sparse(  # pylint: disable=protected-access
        grad, handle, indices)

  def get_name(self):
    return self._wrapped_optimizer.get_name()

  def get_slot(self, var, name):
    return self._wrapped_optimizer.get_slot(var, name)

  def get_slot_names(self):
    return self._wrapped_optimizer.get_slot_names()

  def variables(self):
    return self._wrapped_optimizer.variables()


def AssertAllWeightsReplicaIdentical(test_case, var, replica_count):
  """ Utility for checking that the values logged in KernelLoggingOptimizer are
  replica identical."""
  test_case.assertGreater(len(var), 0, "No weights for variable.")

  for weights in var:
    test_case.assertEqual(
        len(weights), replica_count,
        f"Expected {replica_count} weights but have {len(weights)}.")

    replica1 = weights[0]
    equal_weights = map(lambda x: (replica1 == x).all(), weights[1:])  # pylint: disable=cell-var-from-loop
    test_case.assertTrue(all(equal_weights),
                         "Expected all weights to be replica identical.")
