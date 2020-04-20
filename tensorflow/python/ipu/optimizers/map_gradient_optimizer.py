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
# ==============================================================================


from tensorflow.python.training import optimizer


class MapGradientOptimizer(optimizer.Optimizer):
  """
  This class enables modification of the computed gradients, before they are
  passed to the final optimizer for application.

  MapGradientOptimizer needs a map function that will modify the gradients,
  and an optimizer to which the modified gradients are passed.

  The map function has two arguments: gradient and variable. The map function
  must return the modified gradient.


  Example

  .. code-block:: python

     # Define function which will modify computed gradients.
     # This is a gradient decay function.

     def map_fn_decay(grad, var):
       return grad + (WEIGHT_DECAY * var)

     # To run the code we need a session:
     with self.cached_session():
       optimizer = gradient_descent.GradientDescentOptimizer(0.000001)
       # We define MapGradientOptimizer
       map_optimizer = map_gradient_optimizer.MapGradientOptimizer(
           optimizer, map_fn_decay)
       # Gradients are computed by compute_gradients(), where our map function
       # modifies computed gradients. compute_gradients(loss, var_list) arguments
       # are loss and var_list so define arguments and call
       # map_optimizer.compute_gradients().
       values = [1.0, 2.0, 3.0]
       vars_ = [variables.Variable([v], dtype=dtypes.float32) for v in values]
       grads_and_vars = map_optimizer.compute_gradients(
           vars_[0] * vars_[1] + vars_[0] * vars_[2] + vars_[1] * vars_[2],
           vars_)
       # The output grads_and_vars contains computed gradients modified by
       # the decay map function.
       # grads are 5.01, 4.02 and 3.03. If we did not use MapGradientOptimizer
       # they would be 5, 4 and 3.

  Args:
    wrapped_optimizer: tensorflow (derived) optimizer.
    gradient_mapping_function: is applied on grads and variables which are
      provided by wrapped_optimizer.compute_gradients().

  Returns:
    compute_gradients() returns a list of (gradient, variable) pairs.
  """
  def __init__(self,
               wrapped_optimizer,
               gradient_mapping_function,
               name="MapGradientOptimizer"):

    super(MapGradientOptimizer, self).__init__(False, name)
    self._wrapped_optimizer = wrapped_optimizer
    self._gradient_mapping_function = gradient_mapping_function

  # Override method from tensorflow.python.training.optimizer.Optimizer
  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=optimizer.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    grads_and_vars = self._wrapped_optimizer.compute_gradients(
        loss, var_list, gate_gradients, aggregation_method, True, grad_loss)
    grads_and_vars = [(self._gradient_mapping_function(x[0],
                                                       x[1].value()), x[1])
                      for x in grads_and_vars]
    return grads_and_vars

  # Override method from tensorflow.python.training.optimizer.Optimizer
  def get_name(self):
    return self._wrapped_optimizer.get_name()

  # Override method from tensorflow.python.training.optimizer.Optimizer
  def minimize(self,
               loss,
               global_step=None,
               var_list=None,
               gate_gradients=optimizer.Optimizer.GATE_OP,
               aggregation_method=None,
               colocate_gradients_with_ops=False,
               name=None,
               grad_loss=None):
    return self._wrapped_optimizer.minimize(loss, global_step, var_list,
                                            gat_gradients, aggregation_method,
                                            colocate_gradients_with_ops, name,
                                            grad_loss)

  # Override method from tensorflow.python.training.optimizer.Optimizer
  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    return self._wrapped_optimizer.apply_gradients(grads_and_vars, global_step,
                                                   name)

  # Override method from tensorflow.python.training.optimizer.Optimizer
  def get_slot(self, var, name):
    return self._wrapped_optmizer.get_slot(var, name)

  # Override method from tensorflow.python.training.optimizer.Optimizer
  def get_slot_names(self):
    return self._wrapped_optimizer.get_slot_names()

  # Override method from tensorflow.python.training.optimizer.Optimizer
  def variables(self):
    return self._wrapped_optimizer.variables()
