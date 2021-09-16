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
Optimizer wrapper for automatic loss scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ipu.ops import statistics_ops
from tensorflow.python.ipu.keras.optimizers import IpuOptimizer
from tensorflow.python.ops import array_ops

from tensorflow.python.ipu.optimizers.automatic_loss_scaling_optimizer import AutomaticLossScalingOptimizer as ALS


class AutomaticLossScalingOptimizer(IpuOptimizer):
  """An optimizer that automatically computes and applies
  a loss scaling factor (LSF) prior to gradient computation.

  The LSF is computed such that the magnitude of the loss is increased
  to reduce numerical underflow. If the magnitude of the loss becomes too
  great and overflow occurs, then the LSF is automatically decreased.

  The automatic increase and decrease of the LSF is governed by sample
  statistics collected over computed gradients of type `float16`.

  Gradient statistics are collected on each backward pass, irresepective
  of `update_frequency`. Every `update_frequency` passes, the LSF is
  scaled by either `increase_factor` or `decrease_factor` depending on
  the state of the gradient statistics collected up to that point. If
  there is minimal overflow, then the LSF is scaled by `increase_factor`,
  otherwise it is scaled by `decrease_factor`. At LSF update time, the
  gradient statistics are reset for the following update period.

  Example using Keras Functional API:

  .. code-block:: python
    strategy = IPUStrategy()
    with strategy.scope():
      opt = SGD(0.01)
      opt_wrapper = AutomaticLossScalingOptimizer(
        opt,
        initial_loss_scaling_factor=10.0,
        update_frequency=3,
        increase_factor=2.0,
        decrease_factor=0.5)

      x, t = some_dataset_fn()
      input_l = Input(x.shape[1])

      dense = Dense(t.shape[1], activation='relu', dtype=np.float16)(input_l)

      m = Model(inputs=input_l,
                outputs=dense,
                gradient_accumulation_count=2)
      m.compile(optimizer=opt_wrapper, loss='mse')

      m.fit(x, t)

  Example using `tf.function`:

  .. code-block:: python
    strategy = IPUStrategy()
      opt = SGD(0.01)
      opt_wrapper = AutomaticLossScalingOptimizer(
        opt,
        initial_loss_scaling_factor=10.0,
        update_frequency=3,
        increase_factor=2.0,
        decrease_factor=0.5)

      x, t = some_dataset_fn()

      dense = Dense(t.shape[1], activation='relu', dtype=np.float16)

      @tf.function(experimental_compile=True)
      def f(x, t):
        with GradientTape() as tape:
          y = dense(x)
          l = mean_squared_error(labels=t, predictions=y)

        opt_wrapper.minimize(l, dense.variables, tape=tape)
        return l

      loss = strategy.run(f, args=[x, t])
  """
  def __init__(self,
               opt,
               initial_loss_scaling_factor=1.0,
               update_frequency=8,
               increase_factor=2.0,
               decrease_factor=0.5,
               max_loss_scaling_factor=32768,
               accumulate_statistics_over_update_period=True,
               ratio_threshold=10e-6,
               name="AutomaticLossScalingOptimizer"):
    """Construct a new automatic loss scaling optimizer.

    Args:
      opt: An existing `Optimizer` to encapsulate.
      initial_loss_scaling_factor: The initial Loss Scaling Factor (LSF).
        Defaults to 1.
      update_frequency: The number of steps that should be taken before
        updating the LSF.
        Defaults to 8.
      increase_factor: The factor to scale the LSF by when increasing the LSF.
        Defaults to 2.
      decrease_factor: The factor to scale the LSF by when decreasing the LSF.
        Defaults to 0.5.
      max_loss_scaling_factor: The maximum value to which the LSF can increase.
        Defaults to 32768.
      accumulate_statistics_over_update_period: If true, statistics are
        accumulated over each `update_frequency` period, else they are
        collected once every `update_frequency` updates.
      ratio_threshold: The threshold over which the ratio of overflowed
        float16 gradients to all float16 gradients must exceed to cause a
        reduction in LSF. Ratios not meeting this threshold will cause an
        increase in LSF.
        Defaults to 10e-6.
      name: Optional name prefix for the operation created when applying
        gradients.
        Defaults to "AutomaticLossScalingOptimizer".
    """
    super().__init__(opt, name=name)

    self.update_frequency = update_frequency
    self.increase_factor = increase_factor
    self.decrease_factor = decrease_factor
    self.max_loss_scaling_factor = max_loss_scaling_factor
    self.ratio_threshold = ratio_threshold
    self.initial_loss_scaling_factor = initial_loss_scaling_factor
    self.accumulate_statistics_over_update_period = \
      accumulate_statistics_over_update_period

    self._hist, self._n, self._hist_levels, self._lsf = ALS._get_initial_state(  # pylint: disable=protected-access
        initial_loss_scaling_factor)

  def _assign_var(self, variable, value):
    @def_function.function(experimental_compile=True)
    def f(var, val):
      return var.assign(val)

    return f(variable, value)

  def get_scaled_loss(self, loss):
    """Applies the current loss scaling factor to a given loss.

    Args:
      loss: The loss to be scaled.

    Returns:
      The scaled loss.
    """
    return ALS._get_scaled_loss(loss, self.loss_scaling_factor)  # pylint: disable=protected-access

  def get_unscaled_gradients(self, grads):
    """Collects statistics from LSF scaled gradients and returns the
    same gradients unscaled.

    Args:
      grads: The gradients to be unscaled. These gradients should be
      computed from an LSF scaled loss.

    Returns:
      The unscaled gradients.
    """
    update_hist = ALS._should_update_histogram(  # pylint: disable=protected-access
        self.accumulate_statistics_over_update_period, self.update_counter,
        self.update_frequency)
    grads_unscaled, hist = ALS._get_unscaled_gradients(  # pylint: disable=protected-access
        grads, self.histogram, self.clip_levels, self.loss_scaling_factor,
        update_hist)

    self._assign_var(self._hist, hist)
    return grads_unscaled

  def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
    """Compute gradients of a scaled loss w.r.t. a given list of variables.

    Args:
      loss: A Tensor containing the value to minimize.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKey.TRAINABLE_VARIABLES`.
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    """
    def scaled_loss_fn():
      l = loss() if callable(loss) else loss
      return self.get_scaled_loss(l)

    grads_and_vars = super()._compute_gradients(  # pylint: disable=protected-access
        scaled_loss_fn,
        var_list,
        grad_loss=grad_loss,
        tape=tape)

    grads_and_vars_rescaled = []
    for g, v in grads_and_vars:
      gv = (self.get_unscaled_gradients(g), v)
      grads_and_vars_rescaled.append(gv)

    return grads_and_vars_rescaled

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):  #pylint: disable=arguments-differ
    """Apply gradients to variables and update the loss scale factor.

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
    # Do LSF step.
    lsf, hist, n = ALS._do_lsf_step(  # pylint: disable=protected-access
        self.update_counter, self.update_frequency, self.histogram,
        self.loss_scaling_factor, self.decrease_factor, self.increase_factor,
        self.max_loss_scaling_factor, self.ratio_threshold)
    self._assign_var(self._lsf, lsf)
    self._assign_var(self._hist, hist)
    self._assign_var(self._n, n)

    # Apply grads.
    return super().apply_gradients(grads_and_vars, global_step, name)

  def reset(self):
    """Reset loss scaling."""
    self._assign_var(self._hist, array_ops.zeros_like(self.histogram))
    self._assign_var(self._n, 0)
    self._assign_var(self._lsf, self.initial_loss_scaling_factor)

  def get_config(self):
    """
    Returns the config of the `AutomaticLossScalingOptimizer` instance.
    """
    config = super().get_config()
    config.update({
        'initial_loss_scaling_factor': self.initial_loss_scaling_factor,
        'update_frequency': self.update_frequency,
        'increase_factor': self.increase_factor,
        'decrease_factor': self.decrease_factor,
        'max_loss_scaling_factor': self.max_loss_scaling_factor,
        'accumulate_statistics_over_update_period':
        self.accumulate_statistics_over_update_period,
        'ratio_threshold': self.ratio_threshold
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    """Creates an `AutomaticLossScalingOptimizer` from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same optimizer from the config
    dictionary.

    Arguments:
        config: A Python dictionary, typically the output of get_config.
        custom_objects: A Python dictionary mapping names to additional Python
          objects used to create this optimizer, such as a function used for a
          hyperparameter.

    Returns:
        An `AutomaticLossScalingOptimizer` instance.
    """
    config = config.copy()
    IpuOptimizer._verify_config(config)
    inner_config = config.pop('inner_optimizer_config')
    inner_type = config.pop('inner_optimizer_type')
    inner_opt = inner_type(**inner_config)

    return AutomaticLossScalingOptimizer(inner_opt, **config)

  @property
  def histogram(self):
    return ops.convert_to_tensor(self._hist)

  @histogram.setter
  def histogram(self, _):
    raise ValueError("histogram is a read only property.")

  @property
  def normalized_histogram(self):
    return statistics_ops.histogram_normalize(self.histogram)

  @normalized_histogram.setter
  def normalized_histogram(self, _):
    raise ValueError("normalized_histogram is a read only property.")

  @property
  def loss_scaling_factor(self):
    return ops.convert_to_tensor(self._lsf)

  @loss_scaling_factor.setter
  def loss_scaling_factor(self, _):
    raise ValueError("loss_scaling_factor is a read only property.")

  @property
  def update_counter(self):
    return ops.convert_to_tensor(self._n)

  @update_counter.setter
  def update_counter(self, _):
    raise ValueError("update_counter is a read only property.")

  @property
  def update_frequency(self):
    return self._update_frequency

  @update_frequency.setter
  def update_frequency(self, val):
    ALS._verify_update_frequency(val)  # pylint: disable=protected-access
    self._update_frequency = val

  @property
  def increase_factor(self):
    return self._increase_factor

  @increase_factor.setter
  def increase_factor(self, val):
    dec = self.decrease_factor if hasattr(self, '_decrease_factor') else None
    ALS._verify_increase_factor(val, dec)  # pylint: disable=protected-access
    self._increase_factor = val

  @property
  def decrease_factor(self):
    return self._decrease_factor

  @decrease_factor.setter
  def decrease_factor(self, val):
    inc = self.increase_factor if hasattr(self, '_increase_factor') else None
    ALS._verify_decrease_factor(val, inc)  # pylint: disable=protected-access
    self._decrease_factor = val

  @property
  def clip_levels(self):
    return self._hist_levels

  @clip_levels.setter
  def clip_levels(self, _):
    raise ValueError("clip_levels is a read only property.")

  @property
  def initial_loss_scaling_factor(self):
    return self._initial_lsf

  @initial_loss_scaling_factor.setter
  def initial_loss_scaling_factor(self, val):
    ALS._verify_initial_lsf(  # pylint: disable=protected-access
        val, self.increase_factor, self.max_loss_scaling_factor)
    self._initial_lsf = val

  @property
  def max_loss_scaling_factor(self):
    return self._max_loss_scaling_factor

  @max_loss_scaling_factor.setter
  def max_loss_scaling_factor(self, val):
    ALS._verify_max_lsf(val)  # pylint: disable=protected-access
    self._max_loss_scaling_factor = val

  @property
  def accumulate_statistics_over_update_period(self):
    return self._accumulate_stats

  @accumulate_statistics_over_update_period.setter
  def accumulate_statistics_over_update_period(self, val):
    self._accumulate_stats = val

  @property
  def ratio_threshold(self):
    return self._ratio_threshold

  @ratio_threshold.setter
  def ratio_threshold(self, val):
    ALS._verify_ratio_threshold(val)  # pylint: disable=protected-access
    self._ratio_threshold = val
