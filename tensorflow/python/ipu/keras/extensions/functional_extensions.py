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
IPU specific Keras Functional Model extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import copy

from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu.keras.extensions import model_extensions
from tensorflow.python.training.tracking import base as trackable


class PipelineStage(object):
  """A scope within which Keras Layers and/or calls to Keras layers can be
  assigned to pipeline stages.

  Pipeline stages can be assigned to all calls of layer by constructing the
  layer within a `PipelineStage` scope as follows:

  .. code-block:: python

    strategy = ipu.ipu_strategy.IPUStrategy()
    input_layer = Input(2)
    with strategy.scope():
      with PipelineStage(0):
        x = Dense(4)(input_layer)

      with PipelineStage(1):
        x = Dense(4)(x)

  Pipeline stages can also be assigned to individual layer calls, as follows:

  .. code-block:: python

    strategy = ipu.ipu_strategy.IPUStrategy()
    input_layer = Input(2)
    l = Dense(4)
    with strategy.scope():
      with PipelineStage(0):
        x = l(input_layer)

      with PipelineStage(1):
        x = l(x)

  Pipeline stages assigned to layer calls take precedence over those assigned
  when constructing the layer.
  """
  def __init__(self, stage):
    """Creates a scope within which  Keras Layers and/or calls to Keras layers
    are assigned to pipeline stages.

    Arguments:
      stage: The pipeline stage any Keras Layers created and/or called will be
        assigned to within this scope.
    """
    self._stage = stage

  def __enter__(self):
    if self._stage < 0:
      raise ValueError("%d is not a valid pipeline stage.")

    strategy = ds_context.get_strategy()
    if not isinstance(strategy, ipu_strategy.IPUStrategyV1):
      raise RuntimeError("PipelineStage may only be used from "
                         "within an IPUStrategy context.")

    if hasattr(strategy, "_pipeline_stage"):
      raise RuntimeError("Pipeline stages must not be nested.")

    strategy._pipeline_stage = self._stage  # pylint: disable=protected-access

    return self

  def __exit__(self, _exception_type, _value, _traceback):
    strategy = ds_context.get_strategy()
    assert strategy and hasattr(strategy, "_pipeline_stage")
    delattr(strategy, "_pipeline_stage")


class FunctionalLayerPipelineStageAssignment:
  """A class used to indicate in which pipeline stage an invocation of a layer
  in a `Functional` model should be executed in.

  Keras Layers can be called multiple times in order to share weights between
  layers. Each of these calls produces Tensor output which can be executed in
  different pipeline stages (as long as these stages are mapped to the same
  device).
  """
  def __init__(self, layer, node_index, pipeline_stage=None):
    """Create a new SequentialLayerPipelineStageAssignment.

    Args:
      layer: The Keras layer for which this assignment is for.
      node_index: The specific call to the `layer` that produced a Tensor.
        Layers can be called multiple times in order to share weights. A new
        FunctionalLayerPipelineStageAssignment is required for every Layer call.
        E.g. `node_index=0` will correspond to the first time the `layer` was
        called.
      pipeline_stage: If provided, indicates which pipeline stage this layer
        should be assigned to. If not provided this layer will be unassigned.
    """
    self._layer = layer
    self._node_index = node_index
    self.pipeline_stage = pipeline_stage

  @property
  def layer(self):
    """Returns the Keras layer for which this assignment is for."""
    return self._layer

  @property
  def node_index(self):
    """Returns the specific call to the `layer` that produced a Tensor."""
    return self._node_index

  @property
  def inbound_layers(self):
    """Returns the input layers for the layer in this assignment. This can be
    useful for identifying which specific `node_index` this is."""
    return self._layer._inbound_nodes[self.node_index].inbound_layers  # pylint: disable=protected-access

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
    return ("Layer: {} (node index {}) is assigned to pipeline "
            "stage: {}".format(self.layer.name, self.node_index,
                               self.pipeline_stage))


class FunctionalExtension(model_extensions.ModelExtension):  # pylint: disable=abstract-method
  @trackable.no_automatic_dependency_tracking
  def __init__(self):
    model_extensions.ModelExtension.__init__(self)
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
    # Get index for each layer.
    layer_to_index = {}
    for i, layer in enumerate(self.layers):
      layer_to_index[str(id(layer))] = i
    config["pipeline_stage_assignment"] = [
        (layer_to_index[str(id(assignment.layer))], assignment.node_index,
         assignment.pipeline_stage)
        for assignment in self._pipeline_stage_assignment
    ]
    return config

  def _deserialize_from_config_supported(self, config):
    del config
    return True

  @trackable.no_automatic_dependency_tracking
  def _deserialize_from_config_delegate(self, config):
    FunctionalExtension.__init__(self)
    self._from_base_config(config)
    # Extract pipelining options.
    self._pipeline_stage_assignment = [
        FunctionalLayerPipelineStageAssignment(self.layers[layer_idx],
                                               node_index, stage)
        for layer_idx, node_index, stage in config.get(
            "pipeline_stage_assignment", [])
    ]

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

  @trackable.no_automatic_dependency_tracking
  def _create_post_order(self):
    post_order_node_execution = []
    nodes_by_depth = self._nodes_by_depth
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    visited_set = set()
    for x in self.inputs:
      visited_set.add(str(id(x)))
      post_order_node_execution.append(x)

    for depth in depth_keys:
      nodes = nodes_by_depth[depth]
      for node in nodes:
        if node.is_input:
          # Inputs are handled explicitly.
          continue

        if any(t_id not in visited_set for t_id in node.flat_input_ids):
          # Node is not computable, skip.
          continue

        post_order_node_execution.append(node)

        for x_id in node.flat_output_ids:
          visited_set.add(x_id)

    assert len(post_order_node_execution) == len(self._network_nodes)
    return post_order_node_execution

  @trackable.no_automatic_dependency_tracking
  def _get_pipelined_post_order(self, pipeline_stage_assignment):
    # Create a lookup map for nodes.
    node_to_stage = {}
    found_stages = set()
    for assignment in pipeline_stage_assignment:
      pipeline_stage = assignment.pipeline_stage
      layer = assignment.layer
      if assignment.pipeline_stage is None:
        raise ValueError(
            "Layer {} with node index {} has not been assigned a pipeline "
            "stage.".format(assignment.layer.name, assignment.node_index))
      node = layer._inbound_nodes[assignment.node_index]  # pylint: disable=protected-access
      if str(id(node)) in node_to_stage:
        raise ValueError(
            "Duplicate assignment for layer {} with node index {}. Each layer "
            "invocation can only be assigned to a single pipeline stage.".
            format(assignment.layer.name, assignment.node_index))

      node_to_stage[str(id(node))] = pipeline_stage
      found_stages.add(pipeline_stage)

    assert len(node_to_stage) == (len(self._network_nodes) - len(self.inputs))

    found_stages = sorted(list(found_stages))
    num_stages = found_stages[-1] + 1
    if found_stages != list(range(num_stages)):
      missing_stages = set(range(num_stages)) - set(found_stages)
      raise ValueError(
          "Pipeline stages in the graph need to be strictly increasing, "
          "found pipeline stages %s, however the following pipeline stages "
          "are missing %s." % (", ".join(str(v)
                                         for v in found_stages), ", ".join(
                                             str(v) for v in missing_stages)))

    # Create a post order per pipeline stage as post order does not take
    # pipeline stages into account, for example multiple pipeline stages might
    # have output layers. Try and reorder the the nodes to preserve post order
    # and to make sure pipeline stages can still be executed in order.
    post_order_per_stage = {}
    post_order = self._create_post_order()
    for node in post_order[len(self.inputs):]:
      pipeline_stage = node_to_stage[str(id(node))]
      post_order_per_stage.setdefault(pipeline_stage, []).append(node)

    new_post_order_node_execution = []
    computed_set = set()
    for x in self.inputs:
      layer = x._keras_history.layer  # pylint: disable=protected-access
      assert len(layer.inbound_nodes) == 1
      new_post_order_node_execution.append(layer.inbound_nodes[0])
      computed_set.add(str(id(x)))

    # New post order executes all the layers within a pipeline stage and it
    # makes sure that all the layer inputs have already executed.
    for stage_id in range(num_stages):
      for node in post_order_per_stage[stage_id]:
        all_inputs_executed = all(x in computed_set
                                  for x in node.flat_input_ids)
        if not all_inputs_executed:
          raise ValueError(
              "Layer %s in pipeline stage %d has a dependency from a pipeline "
              "stage which has not yet executed. Layers can only use outputs "
              "from current or previous pipeline stages." %
              (node.outbound_layer.name, stage_id))
        new_post_order_node_execution.append(node)
        # Update computed_set.
        computed_set.update([x for x in node.flat_output_ids])

    return new_post_order_node_execution

  def get_pipeline_stage_assignment(self):
    """Returns the pipeline stage assignment of all the layers in the model.

    If `set_pipeline_stage_assignment()` has been called before, then it returns
    a copy of the current assignment, otherwise returns a list of
    `FunctionalLayerPipelineStageAssignment` for each invocation of each layer
    in the model (excluding input layers).
    """
    if self._pipeline_stage_assignment:
      return copy.copy(self._pipeline_stage_assignment)

    post_order = self._create_post_order()

    output = []
    # Input tensors don't get a pipeline stage so skip them.
    for node in post_order[len(self.inputs):]:
      layer = node.layer
      node_index = layer._inbound_nodes.index(node)  # pylint: disable=protected-access
      output.append(FunctionalLayerPipelineStageAssignment(layer, node_index))

    return output

  def _validate_pipeline_stage_assignment(self, pipeline_stage_assignment):
    # A functional pipeline stage assignment is valid if the graph can be
    # scheduled.
    _ = self._get_pipelined_post_order(pipeline_stage_assignment)

  def _get_pipelining_from_nodes_supported(self):
    return True

  @trackable.no_automatic_dependency_tracking
  def _get_pipelining_from_nodes_delegate(self):
    """Populates pipelining information obtained from users annotating their
    model with `PipelineStage`"""
    post_order = self._create_post_order()

    def node_has_pipeline_stage(node):
      return (hasattr(node, "_pipeline_stage")
              or hasattr(node.outbound_layer, "_pipeline_stage"))

    # Input tensors don't get a pipeline stage so skip them.
    any_node_has_pipeline_stage = any(
        node_has_pipeline_stage(node)
        for node in post_order[len(self.inputs):])

    if not any_node_has_pipeline_stage:
      return

    # If any node has pipelining attached to it, then they all need it.
    # Create pipeline stage assignments.
    pipeline_stage_assignment = []
    for node in post_order[len(self.inputs):]:
      if not hasattr(node, "_pipeline_stage"):
        if not hasattr(node.outbound_layer, "_pipeline_stage"):
          raise ValueError(
              f"All layers of a pipelined model must have an associated "
              f"pipeline stage. However, {node.outbound_layer.name} has not "
              f"been assigned to one. Pipeline stages can be assigned when a "
              f"layer is constructed, or each time a layer is called. "
              f"Different pipeline stages can assigned to each call.")
        node._pipeline_stage = node.outbound_layer._pipeline_stage  # pylint: disable=protected-access
      layer = node.layer
      node_index = layer._inbound_nodes.index(node)  # pylint: disable=protected-access
      pipeline_stage_assignment.append(
          FunctionalLayerPipelineStageAssignment(
              layer, node_index, pipeline_stage=node._pipeline_stage))  # pylint: disable=protected-access

    self._validate_pipeline_stage_assignment(pipeline_stage_assignment)
    self._pipeline_stage_assignment = pipeline_stage_assignment

  @trackable.no_automatic_dependency_tracking
  def set_pipeline_stage_assignment(self, pipeline_stage_assignment):
    """Sets the pipeline stage assignment of all the invocations of all the
    layers in the model.

    Sets the pipeline stage assignment all the invocations of all the
    layers (excluding input layers) in the model which is used to create a
    model-parallel execution of this model when calling `fit()`, `evaluate()`
    and `predict()`. Note that this pipelining stage assignment is ignored when
    using the `call()` function on this model.

    Args:
      pipeline_stage_assignment: A list of the same length as the total number
        of invocations of all the layers in this model (excluding input layers).
        All elements have to be instances of
        `FunctionalLayerPipelineStageAssignment` which are used to indicate
        which pipeline stage a particular layer invocation should be assigned
        to.

    Raises:
      ValueError: `pipeline_stage_assignment` is not a valid assignment.
    """

    if not isinstance(pipeline_stage_assignment, list):
      raise ValueError("`pipeline_stage_assignment` needs to be a list")

    num_invocations = len(self._network_nodes) - len(self.inputs)
    if len(pipeline_stage_assignment) != num_invocations:
      raise ValueError(
          "The size of the provided `pipeline_stage_assignment` ({}) does not "
          "match the total number of invocations of layers in the model "
          "(currently {}). Each invocation of a layer in a model needs to be "
          "assigned to a pipeline stage (excluding input layers).".format(
              len(pipeline_stage_assignment), num_invocations))

    if not all(
        isinstance(assignment, FunctionalLayerPipelineStageAssignment)
        for assignment in pipeline_stage_assignment):
      raise ValueError(
          "All elements of `pipeline_stage_assignment` need to be instances of "
          "`FunctionalLayerPipelineStageAssignment`.")

    self._validate_pipeline_stage_assignment(pipeline_stage_assignment)
    self._pipeline_stage_assignment = pipeline_stage_assignment

    # Pipelining has changed therefore functions need to be recompiled.
    self._reset_ipu_extension()

  @trackable.no_automatic_dependency_tracking
  def reset_pipeline_stage_assignment(self):
    """Resets the pipeline stage assignment so that the model is no longer
    pipelined."""
    self._pipeline_stage_assignment = []

    # Pipelining has changed therefore functions need to be recompiled.
    self._reset_ipu_extension()
