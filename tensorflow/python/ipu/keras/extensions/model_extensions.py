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
IPU specific Keras Model subclass extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from tensorflow.python.ipu.keras.extensions import keras_extension_base
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer
from tensorflow.python.keras.engine import training
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import deprecation


class ModelExtension(keras_extension_base.KerasExtensionBase):  # pylint: disable=abstract-method
  @trackable.no_automatic_dependency_tracking
  def __init__(self):
    keras_extension_base.KerasExtensionBase.__init__(self)
    self._pipeline_stage_assignment_valid = False
    self._pipeline_stage_assignment = []

    # Runtime values
    self._pipeline_maximum_stage = None

  def _is_pipelined(self):
    return bool(self._pipeline_stage_assignment)

  def _get_config_supported(self):
    # Determine if the user has overridden `get_config`, in which case, the
    # model supports `get_config`.
    return self._get_config_overridden()

  def _deserialize_from_config_supported(self, config):
    del config
    # Subclassed models do not support `from_config`.
    return False

  @trackable.no_automatic_dependency_tracking
  def _deserialize_from_config_delegate(self, config):
    ModelExtension.__init__(self)
    self._from_base_config(config)
    # Extract pipelining options.
    self._pipeline_stage_assignment_valid = config.get(
        "pipeline_stage_assignment_valid", False)
    self._pipeline_stage_assignment = [
    ]  # TODO: support pipeline stage assignments

  def set_asynchronous_callbacks(self, asynchronous=False):
    """Sets the asynchronous callbacks options when calling `fit()`, `evaluate()`
    and `predict()`.

    When running `fit()`, `evaluate()` and `predict()` the callbacks the model
    is configured with are executed after `steps_per_execution` have executed.
    Enabling asynchronous callbacks means that the callbacks are invoked after
    every step, even when `steps_per_execution > 1`. This can reduce the latency
    of receiving per step results and metrics at a cost of an extra thread
    running in the background of the application.
    Note that this option is ignored for the `fit()` and `evaluate()` when
    running a pipelined model and `accumulate_outfeed=True` (configured via
    `set_pipelining_options`).

    Args:
      asynchronous: Whether asynchronous callbacks should be enabled.
    """
    self._set_asynchronous_callbacks_impl(asynchronous)

  @deprecation.deprecated_args(
      None, '`experimental_normalize_gradients=True` has been '
      'deprecated and will be replaced in a future release with '
      'the use of mean reduction when accumulating gradients. '
      'Please update your optimizer settings.',
      'experimental_normalize_gradients')
  def set_gradient_accumulation_options(
      self,
      gradient_accumulation_steps_per_replica=None,
      experimental_normalize_gradients=None,
      gradient_accumulation_reduction_method=gradient_accumulation_optimizer.
      GradientAccumulationReductionMethod.SUM,
      **gradient_accumulation_optimizer_kwargs):
    # pylint:disable=line-too-long
    """Sets the gradient accumulation options for non-pipelined models which are
    to be used when training a model.

    When set, and `gradient_accumulation_steps_per_replica > 1`, the optimizer
    which the current model has been compiled with is wrapped in
    :class:`~tensorflow.python.ipu.optimizers.GradientAccumulationOptimizerV2`.
    This means that each replica will accumulate the gradients for
    `gradient_accumulation_steps_per_replica` steps, these accumulated gradients
    are then all-reduced across the replicas and the weight update is performed.

    Gradient Accumulation allows us to simulate bigger batch sizes. For example
    if we have a model where each step is of batch size 16 and we set
    `gradient_accumulation_steps_per_replica=4` and there is single replica in
    the system, this simulates an input batch of size 64.
    If we have a model where each step is of batch size 16 and we set
    `gradient_accumulation_steps_per_replica=4` and there are 4 replicas in
    the system, this simulates an input batch of size 256.

    See the :ref:`gradient-accumulation` section in the documention for more
    details.

    The value of `gradient_accumulation_steps_per_replica` has no effect when
    using `evaluate()` or `predict()`.

    Args:
      gradient_accumulation_steps_per_replica: An integer which indicates the
        number of steps the gradients will be accumulated for in each replica.
        This value multiplied by the number of replicas needs to divide the
        `steps_per_execution` value the model has been compiled with. This value
        is saved/loaded when the model is saved/loaded.
      experimental_normalize_gradients: If set to `True`, the gradients for each
        step are first scaled by
        `1/(gradient_accumulation_steps_per_replica * number of replicas)`
        before being added to the gradient accumulation buffer. Note that this
        option is experimental and the behavior might change in future releases.
        This value is saved/loaded when the model is saved/loaded.
      reduction_method: Reduction method to use when accumulating gradients.
        During the iterations in each optimizer step, the computed gradients
        can either be directly summed up or scaled such that we compute a mean
        of all gradients for each variable. Computing a mean avoids potential
        issues with overflow during accumulation especially when using
        float16, but gives smaller gradients and might require adjusting
        the learning-rate accordingly.
        Defaults to `GradientAccumulationReductionMethod.SUM`
        (see :class:`~tensorflow.python.ipu.optimizers.GradientAccumulationReductionMethod`)  # pylint: disable=line-too-long
      gradient_accumulation_optimizer_kwargs: All remaining keyword arguments
        are forwarded to
        :class:`~tensorflow.python.ipu.optimizers.GradientAccumulationOptimizerV2`.
        See the optimizer for all the available arguments. Must not contain
        `opt` or `num_mini_batches` as keys. Note that this dictionary is not
        serializable, which means that when the model is being saved, these
        values are not saved. When restoring/loading a model, please call
        `set_gradient_accumulation_options` again.
    """
    raise NotImplementedError

  @deprecation.deprecated_args(
      None, '`experimental_normalize_gradients=True` has been '
      'deprecated and will be replaced in a future release with '
      'the use of mean reduction when accumulating gradients. '
      'Please update your pipeline settings.',
      'experimental_normalize_gradients')
  def set_pipelining_options(
      self,
      gradient_accumulation_steps_per_replica=None,
      device_mapping=None,
      accumulate_outfeed=None,
      experimental_normalize_gradients=None,
      gradient_accumulation_reduction_method=gradient_accumulation_optimizer.
      GradientAccumulationReductionMethod.SUM,
      **pipelining_kwargs):
    """Sets the pipelining options, including gradient accumulation options,
    for pipelined models.

    Before training a pipelined model, `gradient_accumulation_steps_per_replica`
    argument needs to be set as pipelined models always perform gradient
    accumulation when training. Setting
    `gradient_accumulation_steps_per_replica > 1` means that each replica will
    accumulate the gradients for `gradient_accumulation_steps_per_replica`
    steps, these accumulated gradients are then all-reduced across the replicas
    and the weight update is performed.

    Gradient Accumulation allows us to simulate bigger batch sizes. For example
    if we have a model where each step is of batch size 16 and we set
    `gradient_accumulation_steps_per_replica=4` and there is single replica in
    the system, this simulates an input batch of size 64.
    If we have a model where each step is of batch size 16 and we set
    `gradient_accumulation_steps_per_replica=4` and there are 4 replicas in
    the system, this simulates an input batch of size 256.

    When training a data-parallel model, enabling gradient accumulation also
    reduces the communication overhead as the all-reduce of gradients is now
    performed after each replica has performed
    `gradient_accumulation_steps_per_replica` steps instead of after each step.

    See the :ref:`gradient-accumulation` section in the documention for more
    details.

    The value of `gradient_accumulation_steps_per_replica` has no effect when
    using `evaluate()` or `predict()`.

    Args:
      gradient_accumulation_steps_per_replica: An integer which indicates the
        number of steps the gradients will be accumulated for in each replica.
        This value multiplied by the number of replicas needs to divide the
        `steps_per_execution` value the model has been compiled with. This value
        is saved/loaded when the model is saved/loaded.
      device_mapping: If provided, a list of length equal to the number of
        pipeline stages assigned in this model. An element at index `i` in the
        list represents which IPU the `i`'th pipeline stage should reside on.
        This can be used to make sure computational stages which share Keras
        layers/`tf.Variable` objects are resident on the same IPU. This value is
        saved/loaded when the model is saved/loaded.
      accumulate_outfeed: The metrics from the model are normally enqueued as
        soon as they're available. If this option is True, the data will
        instead be accumulated when they're available and enqueued at the end of
        pipeline execution, reducing the amount of host <-> device
        communication. When used with training, the accumulated metrics are
        normalised by `gradient_accumulation_steps_per_replica`. When used with
        evaluation, the accumulated metrics are normalised by `steps_per_epoch`.
        This option is ignored when doing prediction. When using
        `accumulate_outfeed`, model callbacks will be called with the same data
        for the batches which the data was accumulated for. This value is
        saved/loaded when the model is saved/loaded.
      experimental_normalize_gradients: If set to `True`, the gradients for each
        step are first scaled by
        `1/(gradient_accumulation_steps_per_replica * number of replicas)`
        before being added to the gradient accumulation buffer. Note that this
        option is experimental and the behavior might change in future releases.
        This value is saved/loaded when the model is saved/loaded.
      reduction_method: Reduction method to use when accumulating gradients.
        During the iterations in each optimizer step, the computed gradients
        can either be directly summed up or scaled such that we compute a mean
        of all gradients for each variable. Computing a mean avoids potential
        issues with overflow during accumulation especially when using
        float16, but gives smaller gradients and might require adjusting
        the learning-rate accordingly.
        Defaults to `GradientAccumulationReductionMethod.SUM`
        (see :class:`~tensorflow.python.ipu.optimizers.GradientAccumulationReductionMethod`)  # pylint: disable=line-too-long
      pipelining_kwargs: All remaining keyword arguments are forwarded to
        :func:`~tensorflow.python.ipu.pipelining_ops.pipeline`. Note that this
        dictionary is not serializable, which means that when the model is
        being saved, these values are not saved. When restoring/loading a model,
        please call `set_pipelining_options` again.
    """
    raise NotImplementedError

  def set_infeed_queue_options(self, **kwargs):
    """Sets the options for all instances of `IPUInfeedQueue` generated
    when executing the model.

    When using `fit()`, `evalute()` and `predict()`, an instance of
    :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue` is created
    to efficiently feed data from the dataset to the device. Instances of
    `IPUInfeedQueue` can be created with optional arguments, such as
    `prefetch_depth`, which can increase the throughput of the model.

    Args:
      **kwargs: All keyword arguments are forwarded to
        :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue`.
    """
    self._set_infeed_queue_options_impl(**kwargs)

  def set_outfeed_queue_options(self, **kwargs):
    """Sets the options for all instances of `IPUOutfeedQueue` generated
    when executing the model.

    When using `fit()`, `evalute()` and `predict()`, an instance of
    :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUOutfeedQueue` is created
    to efficiently feed data from the device to the host. Instances of
    `IPUOutfeedQueue` can be created with optional arguments, such as
    `buffer_depth`, which can increase the throughput of the model.

    Args:
      **kwargs: All keyword arguments are forwarded to
        :class:`~tensorflow.python.ipu.ipu_outfeed_queue.IPUOutfeedQueue`.
    """
    self._set_outfeed_queue_options_impl(**kwargs)

  def get_pipeline_stage_assignment(self):
    raise NotImplementedError

  @trackable.no_automatic_dependency_tracking
  def set_pipeline_stage_assignment(self, pipeline_stage_assignment):
    raise NotImplementedError

  @trackable.no_automatic_dependency_tracking
  def reset_pipeline_stage_assignment(self):
    raise NotImplementedError

  def print_pipeline_stage_assignment_summary(self,
                                              line_length=None,
                                              print_fn=None):
    raise NotImplementedError

  @trackable.no_automatic_dependency_tracking
  def _get_pipeline_maximum_pipeline_stage(self):
    raise NotImplementedError

  def _call_function_overridden(self):
    return True  # the call function *must* be overriden for subclassed `Model`s.

  def _get_config_overridden(self):
    return self.get_config.__func__ != training.Model.get_config
