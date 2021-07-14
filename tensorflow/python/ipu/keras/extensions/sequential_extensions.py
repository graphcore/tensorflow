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
IPU specific Keras Sequentail extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import copy

from tensorflow.python.ipu.keras.extensions import model_extensions
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable


class SequentialLayerPipelineStageAssignment:
  """A class used to indicate in which pipeline stage a layer in a `Sequential`
  model should be executed in.
  """
  def __init__(self, layer, pipeline_stage=None):
    """Create a new SequentialLayerPipelineStageAssignment.

    Args:
      layer: The Keras layer for which this assignment is for.
      pipeline_stage: If provided, indicates which pipeline stage this layer
        should be assigned to. If not provided this layer will be unassigned.
    """
    self._layer = layer
    self.pipeline_stage = pipeline_stage

  @property
  def layer(self):
    """Returns the Keras layer for which this assignment is for."""
    return self._layer

  @property
  def pipeline_stage(self):
    """Returns the pipeline stage this layer has been assigned to. If `None`,
    then this layer has not been assigned to a pipeline stage."""
    return self._pipeline_stage

  @pipeline_stage.setter
  def pipeline_stage(self, value):
    """Setter of `pipeline_stage` property. See `pipeline_stage` property
    doc."""
    self._pipeline_stage = value

  def __str__(self):
    return ("Layer: {} is assigned to pipeline stage: {}".format(
        self.layer.name, self.pipeline_stage))


class SequentialExtension(model_extensions.ModelExtension):  # pylint: disable=abstract-method
  @trackable.no_automatic_dependency_tracking
  def __init__(self):
    model_extensions.ModelExtension.__init__(self)
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
    # Get the ModelExtension config and merge it in.
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
    SequentialExtension.__init__(self)
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

  def set_gradient_accumulation_options(
      self,
      gradient_accumulation_steps=None,
      experimental_normalize_gradients=None,
      **gradient_accumulation_optimizer_kwargs):
    # pylint:disable=line-too-long
    """Sets the gradient accumulation options for non-pipelined models which are
    to be used when training a model.

    When set, and `gradient_accumulation_steps > 1`, the optimizer which the
    current model has been compiled with is wrapped in
    :class:`~tensorflow.python.ipu.optimizers.GradientAccumulationOptimizerV2`.
    This means that instead of performing the weight update for every step,
    gradients across multiple steps are accumulated. After
    `gradient_accumulation_steps` steps have been processed, the accumulated
    gradients are used to compute the weight update.

    Gradient Accumulation allows us to simulate bigger batch sizes. For example
    if we have a model of batch size 16 and we accumulate the gradients for 4
    steps, this simulates an input batch of size 64.

    When training a data-parallel model, enabling gradient accumulation also
    reduces the communication overhead as the all-reduce of gradients is now
    performed every `gradient_accumulation_steps` steps instead of every step.

    See the :ref:`gradient-accumulation` section in the documention for more
    details.

    Args:
      gradient_accumulation_steps: An integer which indicates the number of
        steps the gradients will be accumulated for. This value needs to divide
        the `steps_per_execution` value the model has been compiled with and
        also be divisible by the replication factor if the model is running
        in a data-parallel fashion. This value is saved/loaded when the model
        is saved/loaded.
      experimental_normalize_gradients: If set to `True`, the gradients for each
        step are first scaled by `1/gradient_accumulation_steps` before being
        added to the gradient accumulation buffer. Note that this option is
        experimental and the behavior might change in future releases. This
        value is saved/loaded when the model is saved/loaded.
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
        gradient_accumulation_steps, experimental_normalize_gradients,
        gradient_accumulation_optimizer_kwargs)

  def set_pipelining_options(self,
                             gradient_accumulation_steps=None,
                             device_mapping=None,
                             accumulate_outfeed=None,
                             experimental_normalize_gradients=None,
                             **pipelining_kwargs):
    """Sets the pipelining options, including gradient accumulation options,
    for pipelined models.

    Before training a pipelined model, `gradient_accumulation_steps` argument
    needs to be set as pipelined models always perform gradient accumulation
    when training. Setting `gradient_accumulation_steps > 1` means that instead
    of performing the weight update for every step, gradients across multiple
    steps are accumulated. After `gradient_accumulation_steps` steps have been
    processed, the accumulated gradients are used to compute the weight update.

    Gradient Accumulation allows us to simulate bigger batch sizes. For example
    if we have a model of batch size 16 and we accumulate the gradients for 4
    steps, this simulates an input batch of size 64.

    When training a data-parallel model, enabling gradient accumulation also
    reduces the communication overhead as the all-reduce of gradients is now
    performed every `gradient_accumulation_steps` steps instead of every step.

    See the :ref:`gradient-accumulation` section in the documention for more
    details.

    The value of `gradient_accumulation_steps` has no effect when using
    `evaluate()` or `predict().

    Args:
      gradient_accumulation_steps: An integer which indicates the number of
        steps the gradients will be accumulated for. This value needs to divide
        the `steps_per_execution` value the model has been compiled with and
        also be divisible by the replication factor if the model is running
        in a data-parallel fashion. This value is also constrained on the
        pipelining schedule used. This value is saved/loaded when the model
        is saved/loaded.
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
        normalised `gradient_accumulation_steps`. When used with evaluation, the
        accumulated metrics are normalised by `steps_per_epoch`. This option is
        ignored when doing prediction. When using `accumulate_outfeed`, model
        callbacks will be called with the same data for the batches which the
        data was accumulated for. This value is saved/loaded when the model is
        saved/loaded.
      experimental_normalize_gradients: If set to `True`, the gradients for each
        step are first scaled by `1/gradient_accumulation_steps` before being
        added to the gradient accumulation buffer. Note that this option is
        experimental and the behavior might change in future releases. This
        value is saved/loaded when the model is saved/loaded.
      pipelining_kwargs: All remaining keyword arguments are forwarded to
        :func:`~tensorflow.python.ipu.pipelining_ops.pipeline`. Note that this
        dictionary is not serializable, which means that when the model is
        being saved, these values are not saved. When restoring/loading a model,
        please call `set_pipelining_options` again.
    """
    self._set_pipelining_options_impl(gradient_accumulation_steps,
                                      device_mapping, accumulate_outfeed,
                                      experimental_normalize_gradients,
                                      pipelining_kwargs)

  @trackable.no_automatic_dependency_tracking
  def _get_pipeline_post_order(self, input_shapes, input_dtypes):
    if not self._has_explicit_input_shape:
      # If applicable, update the static input shape of the model.
      if not isinstance(input_shapes, tensor_shape.TensorShape):
        # This is a Sequential with multiple inputs which cannot be pipelined.
        raise RuntimeError(
            "Layers in a Sequential model should only have a single input "
            "tensor, but we received a {} input: {}. Consider rewriting this "
            "model with the Functional API.".format(type(input_shapes),
                                                    input_shapes))
      else:
        self._build_graph_network_for_inferred_shape(input_shapes,
                                                     input_dtypes)

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

  @trackable.no_automatic_dependency_tracking
  def _get_pipeline_maximum_pipeline_stage(self):
    assert self._is_pipelined()
    if self._pipeline_maximum_stage is None:
      self._pipeline_maximum_stage = self._pipeline_stage_assignment[
          -1].pipeline_stage
    return self._pipeline_maximum_stage
