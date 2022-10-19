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
IPU specific Keras Sequential extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import copy

from tensorflow.python.ipu.keras.extensions import keras_extension_base
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import sequential
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import deprecation
from tensorflow.python.util import deprecation


class SequentialLayerPipelineStageAssignment:
  """A class used to indicate which pipeline stage a layer in a `Sequential`
  model should be executed in.
  """
  @deprecation.deprecated(None, "Use `tf.keras` instead of `tf.python.keras`.")
  def __init__(self, layer, pipeline_stage=None):
    """Create a new `SequentialLayerPipelineStageAssignment`.

    Args:
      layer (keras.layers.Layer): The Keras layer for which this assignment is
        for.
      pipeline_stage (int): If provided, indicates which pipeline stage this
        layer should be assigned to. If not provided this layer will be
        unassigned.
    """
    self._layer = layer
    self.pipeline_stage = pipeline_stage

  @property
  def layer(self):
    """Returns the Keras layer associated with this assignment."""
    return self._layer

  @property
  def pipeline_stage(self):
    """Returns the pipeline stage this layer has been assigned to. If `None`,
    then this layer has not been assigned to a pipeline stage."""
    return self._pipeline_stage

  @pipeline_stage.setter
  def pipeline_stage(self, value):
    """Setter of `pipeline_stage` property. See `pipeline_stage` property
    doc.

    Args:
      value (int): The pipeline stage to assign this layer to."""
    self._pipeline_stage = value

  def __str__(self):
    return ("Layer: {} is assigned to pipeline stage: {}".format(
        self.layer.name, self.pipeline_stage))


class SequentialExtension(keras_extension_base.KerasExtensionBase):  # pylint: disable=abstract-method
  @trackable.no_automatic_dependency_tracking
  @deprecation.deprecated(None, "Use `tf.keras` instead of `tf.python.keras`.")
  def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
    keras_extension_base.KerasExtensionBase.__init__(self)
    self._pipeline_stage_assignment_valid = False
    self._pipeline_stage_assignment = []

    # Runtime values
    self._pipeline_maximum_stage = None

  def _is_pipelined(self):
    return bool(self._pipeline_stage_assignment)

  def _get_config_supported(self):
    return True

  def _get_config_delegate(self):
    # Get the Keras config.
    config = self.get_config(__extension_delegate=False)
    # Get the KerasExtensionBase config and merge it in.
    extension_config = self._get_base_config()
    config.update(extension_config)
    # Add pipelining options.
    config["pipeline_stage_assignment_valid"] = \
      self._pipeline_stage_assignment_valid
    config["pipeline_stage_assignment"] = [
        assignment.pipeline_stage
        for assignment in self._pipeline_stage_assignment
    ]

    return config

  def _deserialize_from_config_supported(self, config):
    del config
    return True

  @trackable.no_automatic_dependency_tracking
  def _deserialize_from_config_delegate(self, config):
    self._from_base_config(config)
    # Extract pipelining options.
    self._pipeline_stage_assignment_valid = config.get(
        "pipeline_stage_assignment_valid", False)
    self._pipeline_stage_assignment = [
        SequentialLayerPipelineStageAssignment(self.layers[i], stage)
        for i, stage in enumerate(config.get("pipeline_stage_assignment", []))
    ]

  def _add_supported(self, _):
    return True

  @trackable.no_automatic_dependency_tracking
  def _add_delegate(self, layer):
    # Invalidate pipelining.
    if self._is_pipelined():
      self._pipeline_stage_assignment_valid = False
      logging.info(
          "Adding a layer to a pipelined Sequential model has invalidated the "
          "pipeline stage assignment. You need to call "
          "`set_pipeline_stage_assignment()` before executing again.")

    return self.add(layer, __extension_delegate=False)

  def _pop_supported(self):
    return True

  @trackable.no_automatic_dependency_tracking
  def _pop_delegate(self):
    # Invalidate pipelining.
    if self._is_pipelined():
      self._pipeline_stage_assignment_valid = False
      logging.info(
          "Removing a layer from a pipelined Sequential model has invalidated "
          "the pipeline stage assignment. You need to call "
          "`set_pipeline_stage_assignment()` before executing again.")

    return self.pop(__extension_delegate=False)

  def set_asynchronous_callbacks(self, asynchronous=False):
    """Sets the asynchronous callbacks options when calling `fit()`, `evaluate()`
    and `predict()`.

    When running `fit()`, `evaluate()` and `predict()`, the callback functions
    are called after executing the number of steps specified by
    `steps_per_execution`, where each step processes one batch.

    Enabling asynchronous callbacks means that the callbacks are invoked after
    every step, even when `steps_per_execution > 1`. This can reduce the latency
    of receiving per-step results and metrics, at the cost of an extra thread
    running in the background of the application.

    Note that this option is ignored for `fit()` and `evaluate()` when
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
      gradient_accumulation_reduction_method='sum',
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
        The `steps_per_execution` value used when compiling the model must be
        divisible by `gradient_accumulation_steps_per_replica`. This value is
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
      gradient_accumulation_optimizer_kwargs: All remaining keyword arguments
        are forwarded to
        :class:`~tensorflow.python.ipu.optimizers.GradientAccumulationOptimizerV2`.
        See the optimizer for all the available arguments. Must not contain
        `opt` or `num_mini_batches` as keys. Note that this dictionary is not
        serializable, which means that when the model is being saved, these
        values are not saved. When restoring/loading a model, please call
        `set_gradient_accumulation_options` again.
    """
    # pylint:enable=line-too-long
    self._set_gradient_accumulation_options_impl(
        gradient_accumulation_steps_per_replica,
        experimental_normalize_gradients,
        gradient_accumulation_reduction_method,
        gradient_accumulation_optimizer_kwargs)

  @deprecation.deprecated_args(
      None, '`experimental_normalize_gradients=True` has been '
      'deprecated and will be replaced in a future release with '
      'the use of mean reduction when accumulating gradients. '
      'Please update your pipeline settings.',
      'experimental_normalize_gradients')
  def set_pipelining_options(
      self,  #pylint: disable=missing-type-doc
      gradient_accumulation_steps_per_replica=None,
      device_mapping=None,
      accumulate_outfeed=None,
      experimental_normalize_gradients=None,
      gradient_accumulation_reduction_method='sum',
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
        The `steps_per_execution` value used when compiling the model must be
        divisible by `gradient_accumulation_steps_per_replica`. This value is
        saved/loaded when the model is saved/loaded.
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
      gradient_accumulation_reduction_method: Reduction method to use when
        accumulating gradients. During the iterations in each optimizer step,
        the computed gradients can either be directly summed up or scaled such
        that we compute a mean of all gradients for each variable. Computing a
        mean avoids potential issues with overflow during accumulation
        especially when using float16, but gives smaller gradients and might
        require adjusting the learning-rate accordingly.
        Defaults to `GradientAccumulationReductionMethod.SUM`
        (see :class:`~tensorflow.python.ipu.optimizers.GradientAccumulationReductionMethod`)  # pylint: disable=line-too-long
      pipelining_kwargs: All remaining keyword arguments are forwarded to
        :func:`~tensorflow.python.ipu.pipelining_ops.pipeline`. Note that this
        dictionary is not serializable, which means that when the model is
        being saved, these values are not saved. When restoring/loading a model,
        please call `set_pipelining_options` again.
    """
    self._set_pipelining_options_impl(gradient_accumulation_steps_per_replica,
                                      device_mapping, accumulate_outfeed,
                                      experimental_normalize_gradients,
                                      gradient_accumulation_reduction_method,
                                      pipelining_kwargs)

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

  @trackable.no_automatic_dependency_tracking
  def _build_with_dtypes(self, input_shape, input_dtype):
    if not self._has_explicit_input_shape:
      # If applicable, update the static input shape of the model.
      if not (isinstance(input_shape, tuple)
              and all(d is None or isinstance(d, int) for d in input_shape)):
        # This is a Sequential with multiple inputs which cannot be pipelined.
        raise RuntimeError(
            f"Layers in a Sequential model should only have a single input, "
            f"but we received a {type(input_shape)}: {input_shape}. "
            f"Consider rewriting this model with the Functional API.")
      else:
        self._build_graph_network_for_inferred_shape(input_shape, input_dtype)
    # Bypass the Sequential build method but still call into the base.
    super(sequential.Sequential, self).build(input_shape)  # pylint: disable=bad-super-call

  @trackable.no_automatic_dependency_tracking
  def _get_pipeline_post_order(self):
    if not self._graph_initialized:
      raise RuntimeError(
          "The Sequential model {} cannot be represented as a graph network, "
          "this could be because:\n * A layer in your model failed to "
          "evaluate.\n * The layer is dynamic and therefore not graph "
          "compatible.".format(self.name))

    if not self.built:
      self._init_graph_network(self.inputs, self.outputs)

    post_order = self._create_post_order()
    num_inputs = len(self.inputs)
    assert (len(self._pipeline_stage_assignment) +
            num_inputs) == len(post_order)

    nodes_per_stage = {}
    for idx, node in enumerate(post_order):
      if idx < num_inputs:
        layer = node._keras_history.layer  # pylint: disable=protected-access
        assert len(layer.inbound_nodes) == 1
        nodes_per_stage.setdefault(0, []).append(layer.inbound_nodes[0])
      else:
        assignment = self._pipeline_stage_assignment[idx - num_inputs]
        assert node.layer is assignment.layer
        nodes_per_stage.setdefault(assignment.pipeline_stage, []).append(node)
    return nodes_per_stage

  def get_pipeline_stage_assignment(self):
    """Returns the pipeline stage assignment of the layers in the model.

    If `set_pipeline_stage_assignment()` has been called before, then it returns
    a copy of the current assignment, otherwise returns a list of
    `SequentialLayerPipelineStageAssignment` for each layer in the model in
    post order (which means that layers are returned in the order they are
    executed).
    """
    if self._pipeline_stage_assignment:
      if not self._pipeline_stage_assignment_valid:
        logging.info(
            "Calling `get_pipeline_stage_assignment()` on a model which has "
            "had layers added/removed since the last "
            "`set_pipeline_stage_assignment()` call which means that the "
            "current assignment is not valid.")
      return copy.copy(self._pipeline_stage_assignment)

    return [
        SequentialLayerPipelineStageAssignment(layer) for layer in self.layers
    ]

  def _validate_pipeline_stage_assignment(self, pipeline_stage_assignment):
    # Pipeline stages need to be strictly increasing.
    prev_pipeline_stage = 0
    for i, assignment in enumerate(pipeline_stage_assignment):
      if assignment.pipeline_stage is None:
        raise ValueError(
            "Layer {} has not been assigned a pipeline stage.".format(
                assignment.layer.name))

      if self.layers[i] != assignment.layer:
        raise ValueError(
            "The provided assignment at index {} `pipeline_stage_assignment` "
            "is for layer {}, but the layer in the Sequential model at index "
            "{} is {}.".format(i, assignment.layer.name, i,
                               self.layers[i].name))

      if i == 0:
        if assignment.pipeline_stage != 0:
          raise ValueError(
              "The first layer in a pipelined sequential model needs to be "
              "assigned to the 0th pipeline stage, however it was assigned to "
              "{}.".format(assignment.pipeline_stage))
      elif not assignment.pipeline_stage in [
          prev_pipeline_stage, prev_pipeline_stage + 1
      ]:
        raise ValueError(
            "Layer {} has been assigned to pipeline stage {}, however the "
            "previous layer in the Sequential model was assigned to pipeline "
            "stage {}. A layer in a Sequential model can only be assigned to "
            "the same pipeline stage as the previous layer or to the next "
            "pipeline stage.".format(assignment.layer.name,
                                     assignment.pipeline_stage,
                                     prev_pipeline_stage))

      prev_pipeline_stage = assignment.pipeline_stage

  @trackable.no_automatic_dependency_tracking
  def set_pipeline_stage_assignment(self, pipeline_stage_assignment):
    """Sets the pipeline stage assignment of all the layers in the model.

    Sets the pipeline stage assignment of all the layers in the model which is
    used to create a model-parallel execution of this `Sequential` model when
    calling `fit()`, `evaluate()` and `predict()`. Note that this pipelining
    stage assignment is ignored when using the `call()` function on this model.

    Args:
      pipeline_stage_assignment: A list of the same length as the number of
        layers in this model. All elements can be either intergers or instances
        of `SequentialLayerPipelineStageAssignment`. If all the elements are
        integers, then a layer in this model at index `i` is assigned to a
        pipeline stage `pipeline_stage_assignment[i]`. Otherwise, if all the
        elements are of type `SequentialLayerPipelineStageAssignment` then a
        layer in this model at index `i` is assigned to a pipeline stage
        indicated by `pipeline_stage_assignment[i].pipeline_stage`.

    Raises:
      ValueError: `pipeline_stage_assignment` is not a valid assignment.
    """
    if not isinstance(pipeline_stage_assignment, list):
      raise ValueError("`pipeline_stage_assignment` needs to be a list")

    if len(pipeline_stage_assignment) != len(self.layers):
      raise ValueError(
          "The size of the provided `pipeline_stage_assignment` ({}) does not "
          "match the number of layers in the model (currently {}). Each layer "
          "in a Sequential model needs to be assigned to a pipeline "
          "stage.".format(len(pipeline_stage_assignment), len(self.layers)))

    if all(
        isinstance(assignment, int)
        for assignment in pipeline_stage_assignment):
      # Convert the assignment to `SequentialLayerPipelineStageAssignment`.
      pipeline_stage_assignment = [
          SequentialLayerPipelineStageAssignment(self.layers[i], stage)
          for i, stage in enumerate(pipeline_stage_assignment)
      ]

    if not all(
        isinstance(assignment, SequentialLayerPipelineStageAssignment)
        for assignment in pipeline_stage_assignment):
      raise ValueError(
          "All elements of `pipeline_stage_assignment` need to be instances of "
          "either `int` or `SequentialLayerPipelineStageAssignment`.")

    self._validate_pipeline_stage_assignment(pipeline_stage_assignment)
    self._pipeline_stage_assignment_valid = True
    self._pipeline_stage_assignment = pipeline_stage_assignment
    self._pipeline_maximum_stage = None

    # Pipelining has changed therefore functions need to be recompiled.
    self._reset_ipu_extension()

  @trackable.no_automatic_dependency_tracking
  def reset_pipeline_stage_assignment(self):
    """Resets the pipeline stage assignment so that the model is no longer
    pipelined."""
    self._pipeline_stage_assignment_valid = False
    self._pipeline_stage_assignment = []
    self._pipeline_maximum_stage = None

    # Pipelining has changed therefore functions need to be recompiled.
    self._reset_ipu_extension()

  def print_pipeline_stage_assignment_summary(self,
                                              line_length=None,
                                              print_fn=None):
    """Prints a summary of the pipeline stage assignment of the model.

    Arguments:
        line_length: Total length of printed lines (e.g. set this to adapt the
          display to different terminal window sizes).
        print_fn: Print function to use. It will be called on each line of the
          summary. You can set it to a custom function in order to capture the
          string summary. It defaults to `print` (prints to stdout).
    """
    line_length = line_length or 60

    def print_assignment_fn(assignment, print_row):
      layer = assignment.layer
      pipeline_stage = str(assignment.pipeline_stage)

      name = layer.name
      cls_name = layer.__class__.__name__

      fields = [name + ' (' + cls_name + ')', pipeline_stage]
      print_row(fields)

    headers = ['Layer (type)', 'Pipeline Stage']
    column_widths = [.5, 1.]
    self._print_pipeline_stage_assignment_summary_impl(print_assignment_fn,
                                                       headers, column_widths,
                                                       line_length, print_fn)

  @trackable.no_automatic_dependency_tracking
  def _get_pipeline_maximum_pipeline_stage(self):
    assert self._is_pipelined()
    if self._pipeline_maximum_stage is None:
      self._pipeline_maximum_stage = self._pipeline_stage_assignment[
          -1].pipeline_stage
    return self._pipeline_maximum_stage

  def _validate_call_function(self):
    call_function_overridden = hasattr(self.call, "__func__") and \
                               self.call.__func__ != sequential.Sequential.call
    if call_function_overridden and self._is_pipelined():
      raise RuntimeError(
          f"The function `call` for the model {self.name} has been overridden. "
          f"This is not supported for pipelined Keras Sequential models.")
