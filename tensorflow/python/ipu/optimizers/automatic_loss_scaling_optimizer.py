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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ipu.ops import statistics_ops
from tensorflow.python.ipu.optimizers import IpuOptimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables


class AutomaticLossScalingOptimizer(IpuOptimizer):
  def __init__(self,
               wrapped_optimizer,
               initial_loss_scaling_factor=1.0,
               update_frequency=8,
               increase_factor=2.0,
               decrease_factor=0.5,
               max_loss_scaling_factor=32768,
               accumulate_statistics_over_update_period=True,
               ratio_threshold=10e-6,
               name="AutomaticLossScalingOptimizer"):
    """ Construct an AutomaticLossScalingOptimizer.

    Args:
      wrapped_optimizer: TensorFlow (derived) optimizer.
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
        Defaults to True.
      ratio_threshold: The threshold over which the ratio of overflowed
        float16 gradients to all float16 gradients must exceed to cause a
        reduction in LSF. Ratios not meeting this threshold will cause an
        increase in LSF.
        Defaults to 10e-6.
    """
    super().__init__(wrapped_optimizer, name=name)

    # Store config.
    self.initial_loss_scaling_factor = initial_loss_scaling_factor
    self.update_frequency = update_frequency
    self.increase_factor = increase_factor
    self.decrease_factor = decrease_factor
    self.max_loss_scaling_factor = max_loss_scaling_factor
    self.ratio_threshold = ratio_threshold
    self.accumulate_statistics_over_update_period = \
      accumulate_statistics_over_update_period

    # Verify that the configuration is sane.
    self._verify_update_frequency(self.update_frequency)
    self._verify_increase_factor(self.increase_factor, self.decrease_factor)
    self._verify_decrease_factor(self.decrease_factor, self.increase_factor)
    self._verify_max_lsf(self.max_loss_scaling_factor)
    self._verify_ratio_threshold(ratio_threshold)
    self._verify_initial_lsf(self.initial_loss_scaling_factor,
                             self.increase_factor,
                             self.max_loss_scaling_factor)

    # Initial variables.
    self.histogram, self._n, self.histogram_levels, self.loss_scaling_factor = \
      self._get_initial_state(initial_loss_scaling_factor)

  def compute_gradients(self, loss, **kwargs):  # pylint: disable=arguments-differ
    """ Compute gradients of "loss" scaled by the Loss Scaling Factor
    for the variables in "var_list".

    Args:
      loss: A Tensor containing the value to minimize.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKey.TRAINABLE_VARIABLES`.
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    """
    scaled_loss = self._get_scaled_loss(loss, self.loss_scaling_factor)
    grads_and_vars = self._opt.compute_gradients(scaled_loss, **kwargs)

    update_hist = self._should_update_histogram(
        self.accumulate_statistics_over_update_period, self._n,
        self.update_frequency)

    hist = ops.convert_to_tensor(self.histogram)
    grads_and_vars_rescaled = []
    for g, v in grads_and_vars:
      g_unscaled, hist = self._get_unscaled_gradients(g, hist,
                                                      self.histogram_levels,
                                                      self.loss_scaling_factor,
                                                      update_hist)

      grads_and_vars_rescaled.append((g_unscaled, v))

    self.histogram = self.histogram.assign(hist)

    return grads_and_vars_rescaled

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    # Do LSF step.
    lsf, hist, n = self._do_lsf_step(self._n, self.update_frequency,
                                     self.histogram, self.loss_scaling_factor,
                                     self.decrease_factor,
                                     self.increase_factor,
                                     self.max_loss_scaling_factor,
                                     self.ratio_threshold)
    self.loss_scaling_factor = self.loss_scaling_factor.assign(lsf)
    self.histogram = self.histogram.assign(hist)
    self._n = self._n.assign(n)

    return self._opt.apply_gradients(grads_and_vars,
                                     global_step=global_step,
                                     name=name)

  @staticmethod
  def _get_updated_lsf(histogram, loss_scaling_factor, decrease_factor,
                       increase_factor, max_lsf, ratio_threshold):
    ratio = histogram[1] / math_ops.reduce_sum(histogram)
    lsf = control_flow_ops.cond(math_ops.greater(ratio, ratio_threshold),
                                lambda: loss_scaling_factor * decrease_factor,
                                lambda: loss_scaling_factor * increase_factor)

    # Check the lsf hasn't over or under flowed.
    lsf = control_flow_ops.cond(math_ops.is_finite(lsf), lambda: lsf,
                                lambda: loss_scaling_factor)

    # Check the lsf hasn't exceeded the maximum value.
    lsf = control_flow_ops.cond(math_ops.less(lsf, max_lsf), lambda: lsf,
                                lambda: loss_scaling_factor)

    # Check that lsf >= 1
    return control_flow_ops.cond(math_ops.greater_equal(lsf, 1.0), lambda: lsf,
                                 lambda: loss_scaling_factor)

  @staticmethod
  def _get_scaled_loss(loss, loss_scaling_factor):
    # Get as tensors, these may be variables.
    lsf = math_ops.cast(loss_scaling_factor, loss.dtype)
    return lsf * loss

  @staticmethod
  def _lsf_update_due(update_counter, update_frequency):
    return math_ops.equal(math_ops.floormod(update_counter, update_frequency),
                          0)

  @staticmethod
  def _should_update_histogram(accumulate_statistics, update_counter,
                               update_frequency):
    if accumulate_statistics:
      return True

    return AutomaticLossScalingOptimizer._lsf_update_due(
        update_counter, update_frequency)

  @staticmethod
  def _get_unscaled_gradients(grads, histogram, clip_levels,
                              loss_scaling_factor, update_hist):
    # Get as tensors, these may be variables.
    loss_scaling_factor = ops.convert_to_tensor(loss_scaling_factor)
    clip_levels = ops.convert_to_tensor(clip_levels)
    hist = ops.convert_to_tensor(histogram)
    update_hist = ops.convert_to_tensor(update_hist)

    grads_rescaled = []

    def get_updated_hist(g, h):
      if g.dtype != dtypes.float16:
        return h

      g32 = array_ops.reshape(math_ops.cast(g, dtypes.float32), [-1])
      return statistics_ops.histogram_update(h,
                                             g32,
                                             clip_levels,
                                             absolute_of_input=True)

    def do_update_and_rescale(g, h, rescaled):
      # Add grads to histogram.
      h = control_flow_ops.cond(update_hist, lambda: get_updated_hist(g, h),
                                lambda: h)

      # Rescale grads.
      g_rescaled = g / math_ops.cast(loss_scaling_factor, g.dtype)

      rescaled.append(g_rescaled)
      return h

    is_list = isinstance(grads, list)
    if is_list:
      for g in grads:
        hist = do_update_and_rescale(g, hist, grads_rescaled)
    else:
      hist = do_update_and_rescale(grads, hist, grads_rescaled)

    grads_out = grads_rescaled if is_list else grads_rescaled[0]
    return grads_out, hist

  @staticmethod
  def _do_lsf_step(update_counter, update_frequency, histogram,
                   loss_scaling_factor, decrease_factor, increase_factor,
                   max_lsf, ratio_threshold):
    # Are we due an LSF update?
    do_lsf_update = AutomaticLossScalingOptimizer._lsf_update_due(
        update_counter, update_frequency)

    # Get the latest LSF.
    lsf = control_flow_ops.cond(
        do_lsf_update,
        lambda: AutomaticLossScalingOptimizer._get_updated_lsf(
            histogram, loss_scaling_factor, decrease_factor, increase_factor,
            max_lsf, ratio_threshold),  # pylint: disable=W0108
        lambda: loss_scaling_factor)

    # Reset the gradient histogram if we have performed an LSF update.
    hist = control_flow_ops.cond(do_lsf_update,
                                 lambda: array_ops.zeros_like(histogram),
                                 lambda: histogram)

    # Update counter.
    n = control_flow_ops.cond(do_lsf_update, lambda: 1,
                              lambda: update_counter + 1)

    return lsf, hist, n

  @staticmethod
  def _get_initial_state(initial_loss_scaling_factor):
    # Start with no collected stats.
    hist = variables.Variable(initial_value=array_ops.zeros(
        2, dtype=dtypes.float32),
                              trainable=False,
                              dtype=dtypes.float32,
                              name="gradient_histogram")

    # Counter for LSF update.
    n = variables.Variable(initial_value=1,
                           trainable=False,
                           dtype=dtypes.int32,
                           name="lsf_update_counter")

    # We have two histogram bins, each corresponding to a numerical state;
    # ok and overflow. As such, the binning of gradients is based
    # on the numerical extrema of the float16 representable range.
    hist_levels = constant_op.constant([dtypes.float16.max - 2 * K.epsilon()],
                                       dtype=dtypes.float32)

    lsf = variables.Variable(initial_value=initial_loss_scaling_factor,
                             trainable=False,
                             dtype=dtypes.float32,
                             name="loss_scaling_factor")

    return hist, n, hist_levels, lsf

  @staticmethod
  def _verify_initial_lsf(lsf, increase_factor, max_lsf):
    if lsf <= 0:
      raise ValueError(
          "initial_loss_scaling_factor must be nonzero and positive")

    if lsf >= max_lsf:
      raise ValueError("initial_loss_scaling_factor must be less "
                       "than max_loss_scaling_factor")

    if lsf * increase_factor >= max_lsf:
      raise ValueError(
          "initial_loss_scaling_factor x increase_factor must be less "
          "than max_loss_scaling_factor")

  @staticmethod
  def _verify_decrease_factor(decrease_factor, increase_factor=None):
    if decrease_factor <= 0:
      raise ValueError("decrease_factor must be nonzero and positive")

    if increase_factor and decrease_factor >= increase_factor:
      raise ValueError("decrease_factor must be less than increase_factor")

  @staticmethod
  def _verify_increase_factor(increase_factor, decrease_factor=None):
    if increase_factor <= 0:
      raise ValueError("increase_factor must be nonzero and positive")

    if decrease_factor and increase_factor <= decrease_factor:
      raise ValueError("increase_factor must be greater than decrease_factor")

  @staticmethod
  def _verify_update_frequency(update_frequency):
    if update_frequency <= 0:
      raise ValueError("update_frequency must be nonzero and positive")

  @staticmethod
  def _verify_max_lsf(max_lsf):
    if max_lsf <= 1:
      raise ValueError("max_loss_scaling_factor must be greater than one")

  @staticmethod
  def _verify_ratio_threshold(ratio_threshold):
    if ratio_threshold >= 1.0 or ratio_threshold <= 0.0:
      raise ValueError(
          "ratio_threshold must be greater than zero and less than one")
