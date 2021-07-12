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
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable


class SequentialLayerPipelineStageAssignment:
  """A class used to indicate in which pipeline stage a layer in a `Sequential`
  model should be executed in.
  """
  def __init__(self, model, layer_index, pipeline_stage=None):
    """Create a new SequentialLayerPipelineStageAssignment.

    Args:
      model: The Keras Sequential model which is being pipelined.
      layer_index: The index in the sequence of layers in the `model` for which
        this assignment is for.
      pipeline_stage: If provided, indicates which pipeline stage this layer
        should be assigned to. If not provided this layer will be unassigned.
    """
    self._model = model
    self._layer_index = layer_index
    self.pipeline_stage = pipeline_stage

  @property
  def layer(self):
    """Returns the Keras layer for which this assignment is for."""
    return self._model.layers[self.layer_index]

  @property
  def layer_index(self):
    """Returns the index in the sequence of layers in the Sequential model for
    which this assignment is for."""
    return self._layer_index

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
    return ("Layer: {} (Sequential layer index {}) is assigned to pipeline "
            "stage: {}".format(self.layer.name, self.layer_index,
                               self.pipeline_stage))


class SequentialExtension(model_extensions.ModelExtension):  # pylint: disable=abstract-method
  @trackable.no_automatic_dependency_tracking
  def __init__(self):
    model_extensions.ModelExtension.__init__(self)
    self._pipeline_stage_assignment_valid = False
    self._pipeline_stage_assignment = []

  def _get_shard_count(self):
    return 1

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
        SequentialLayerPipelineStageAssignment(self, i, stage)
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

    This feature of neural networks allows us to simulate bigger batch sizes.
    For example if we have a model of batch size 16 and we accumulate the
    gradients for 4 steps, this simulates an input batch of size 64.

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

  def get_pipeline_stage_assignment(self):
    """Returns the pipeline stage assignment of the layers in the model.

    If `set_pipeline_stage_assignment()` has been called before, then it returns
    a copy of the current assignment, otherwise returns a list of
    `SequentialLayerPipelineStageAssignment` for each layer in the model.
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
        SequentialLayerPipelineStageAssignment(self, i)
        for i in range(len(self.layers))
    ]

  def _validate_pipeline_stage_assignment(self, pipeline_stage_assignment):
    num_layers = len(self.layers)

    if len(pipeline_stage_assignment) != num_layers:
      raise ValueError(
          "The size of the provided `pipeline_stage_assignment` ({}) does not "
          "match the number of layers in the model (currently {}). Each layer "
          "in a Sequential model needs to be assigned to a pipeline "
          "stage.".format(len(pipeline_stage_assignment), num_layers))

    layers_assigned = set()
    for assignment in pipeline_stage_assignment:
      if assignment.layer_index in layers_assigned:
        raise ValueError(
            "The provided `pipeline_stage_assignment` contains a duplicate "
            "assignment for layer {} at sequential index {}. Each layer index "
            "in a Sequential model can only be assigned to a pipeline "
            "stage.".format(assignment.layer.name, assignment.layer_index))
      layers_assigned.add(assignment.layer_index)

    # Pipeline stages need to be strictly increasing.
    prev_pipeline_stage = 0
    for i, assignment in enumerate(pipeline_stage_assignment):
      if i != assignment.layer_index:
        raise ValueError(
            "The provided assignment at index {} `pipeline_stage_assignment` "
            "does not match with the layer index {}".format(
                i, assignment.layer_index))

      if assignment.pipeline_stage is None:
        raise ValueError(
            "Layer {} at sequential index {} has not been assigned a pipeline "
            "stage.".format(assignment.layer.name, assignment.layer_index))

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
            "Layer {} at sequential index {} has been assigned to pipeline "
            "stage {}, however the previous layer in the Sequential model was "
            "assigned to pipeline stage {}. A layer in a Sequential "
            "model can only be assigned to the same pipeline stage as the "
            "previous layer or to the next pipeline stage.".format(
                assignment.layer.name, assignment.layer_index,
                assignment.pipeline_stage, prev_pipeline_stage))

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

    if all(
        isinstance(assignment, int)
        for assignment in pipeline_stage_assignment):
      # Convert the assignment to `SequentialLayerPipelineStageAssignment`.
      pipeline_stage_assignment = [
          SequentialLayerPipelineStageAssignment(self, i, stage)
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

    # Pipelining has changed therefore functions need to be recompiled.
    self._reset_ipu_extension()

  @trackable.no_automatic_dependency_tracking
  def reset_pipeline_stage_assignment(self):
    """Resets the pipeline stage assignment so that the model is no longer
    pipelined."""
    self._pipeline_stage_assignment_valid = False
    self._pipeline_stage_assignment = []

    # Pipelining has changed therefore functions need to be recompiled.
    self._reset_ipu_extension()
