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
IPU specific Keras Model extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import copy
import enum
import math
import sys
import six
import libpvti
import popdist

from collections import OrderedDict
from functools import partial

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device as tf_device
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ipu.keras.extensions import data_adapter as ipu_data_adapter
from tensorflow.python.ipu.keras.extensions import keras_polling_thread
from tensorflow.python.ipu.keras.extensions import keras_util
from tensorflow.python.ipu.keras.extensions import keras_data_feed_manager
from tensorflow.python.ipu.keras import optimizers as ipu_optimizers
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine import training as training_module
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc

logged_steps_per_execution_warning = False
_pvti_trace_channel = libpvti.createTraceChannel("Keras")


class _NormalizedKerasOptimizerWrapper(  # pylint: disable=abstract-method
    ipu_optimizers._KerasOptimizerWrapper):  # pylint: disable=protected-access
  def __init__(self, normalize, normalization_factor, model, optimizer):
    super().__init__(model, optimizer)
    self._normalize = normalize
    self._normalization_factor = normalization_factor

  def preprocess_gradients(self, x):
    grad, var = x
    if self._normalize and grad is not None:
      grad = grad * array_ops.constant(1.0 / self._normalization_factor,
                                       dtype=grad.dtype)
    return grad, var


class _Mode(enum.Enum):
  FIT = 1
  EVALUATE = 2
  PREDICT = 3


class KerasExtensionBase(base_layer.KerasExtension):
  @trackable.no_automatic_dependency_tracking
  def __init__(self):
    # Following values need to be serializable.

    # Pipelining.
    self._pipelining_gradient_accumulation_steps_per_replica = None
    self._pipelining_device_mapping = None
    self._pipelining_accumulate_outfeed = None
    self._pipelining_kwargs = dict()
    self._experimental_pipelining_normalize_gradients = None

    # Gradient accumulation.
    self._gradient_accumulation_steps_per_replica = None
    self._gradient_accumulation_optimizer_kwargs = dict()
    self._gradient_accumulation_reduction_method = \
      gradient_accumulation_optimizer.GradientAccumulationReductionMethod.SUM
    self._experimental_gradient_accumulation_normalize_gradients = None

    # Asynchronous callbacks.
    self._asynchronous_callbacks = False

    # Datafeed managers.
    self._infeed_kwargs = dict()
    self._outfeed_kwargs = dict()

    # Following values are runtime only.
    self._use_synthetic_data = utils.use_synthetic_data_for(
        utils.SyntheticDataCategory.Outfeed)
    self._compiled_gradient_accumulation_steps_per_replica = None
    self._compiled_pipeline_gradient_accumulation_steps_per_replica = None
    self._compiled_pipeline_train_iterations = None
    self._compiled_pipeline_test_iterations = None
    self._compiled_pipeline_predict_iterations = None

    self._reset_ipu_extension()

  def _log_steps_per_execution_warning(self, steps_per_execution):
    """If `steps_per_execution = 1`, a warning is logged so the user
    is notified that additional performance can be gained by increasing
    `steps_per_execution`.

    Args:
        steps_per_execution (integer): Number of steps to compile in
        one Poplar program.
    """
    global logged_steps_per_execution_warning
    if steps_per_execution == 1:
      logging.info("The model `{}` has been configured with only {} steps per "
                   "execution. Consider increasing the value for the "
                   "`steps_per_execution` argument passed to the `compile()` "
                   "method to improve performance.".format(
                       self.name, steps_per_execution))
      logged_steps_per_execution_warning = True

  def _log_optimizer_batch_size(self, data_handler):
    """A function that logs the batch size as seen from the perspective of the
    optimizer during training.

    Args:
        data_handler (IPUDataHandler): The data handler created in `fit()`
    """
    # Optimizer batch size depends on the specified batch size, the gradient
    # accumulation and the replication factor.
    steps_per_execution = data_handler.steps_per_execution_value
    gradient_accumulation_steps_per_replica = \
      self._verify_and_get_gradient_accumulation_steps_per_replica(
          steps_per_execution)
    total_replicas = self._get_replication_factor() * popdist.getNumInstances()
    # Construct tailored message depending on if replication, gradient
    # accunulation, or both are enabled.
    is_distributed = total_replicas > 1
    is_accumulated = gradient_accumulation_steps_per_replica > 1
    if is_accumulated or is_distributed:
      accumulating_n_batches = \
        " and accumulating {} batches per optimizer step".format(
            gradient_accumulation_steps_per_replica)
      across_n_replicas = " across {} replicas".format(total_replicas)
      effective_batch_size = data_handler.batch_size * \
        gradient_accumulation_steps_per_replica * total_replicas
      logging.info(
          "Training is{}{}{}, your effective batch size is {}.".format(
              " distributed" if is_distributed else "",
              accumulating_n_batches if is_accumulated else "",
              across_n_replicas if is_distributed else "",
              effective_batch_size))

  def _get_shard_count(self):
    """Returns how many shards the model is parallelized over.

    Returns:
        integer: Number of shards.
    """
    if self._is_pipelined():
      if self._pipelining_device_mapping:
        return max(self._pipelining_device_mapping) + 1
      return self._get_pipeline_maximum_pipeline_stage() + 1
    return 1

  def _is_pipelined(self):
    """Returns whether the model is pipelined or not.

    Raises:
        NotImplementedError: This is the base class and this method needs
        to be implemented in the extended classes.
    """
    raise NotImplementedError

  def _check_mode(self):
    """Asserts that the mode that we run the model in is supported on the IPU.

    Raises:
        RuntimeError: When the model is in eager mode.
    """
    if self.run_eagerly:
      raise RuntimeError(
          "Keras models cannot run eagerly when using `IPUStrategy`. Set "
          "`run_eagerly=False` when calling `compile`.")

  def _get_num_ipus(self):
    """Returns the number of physical IPUs in the current tf.Device.

    Raises:
        ValueError: When the current TF device is not an IPU.

    Returns:
        integer: Number of physical IPUs in the current TF device.
    """
    device_string = self.distribute_strategy.extended.non_slot_devices(None)
    current_device = tf_device.DeviceSpec.from_string(device_string)

    if current_device.device_type != "IPU":
      raise ValueError(self.__class__.__name__ +
                       " can only be used on an IPU device.")

    # get_num_of_ipus_in_device() returns the number of devices for all instances combined.
    return int(
        utils.get_num_of_ipus_in_device(device_string) /
        popdist.getNumInstances())

  def _get_replication_factor(self):
    """Calculate the replication factor of the current model. This is calculated
    by `num_ipus / shard_count`.

    Raises:
        ValueError: When the `shard_count` is greater than the number of
        physical IPUs.

    Returns:
        integer: The replication factor of the current model.
    """
    if self._replication_factor is None:
      num_ipus = self._get_num_ipus()
      shard_count = 2**int(math.ceil(math.log2(self._get_shard_count())))

      if self._get_shard_count() > num_ipus:
        raise ValueError(
            "Current device has {} IPUs attached, however the current model "
            "requires a multiple of {} IPUs.".format(num_ipus, shard_count))

      self._replication_factor = int(num_ipus // shard_count)

    return self._replication_factor

  def _verify_and_get_gradient_accumulation_steps_per_replica(
      self, steps_per_execution):
    """Verifies the number of steps necessary for the defined gradient
    accumulation settings.

    Args:
        steps_per_execution (integer): Number of steps to compile in
        one Poplar program.

    Raises:
        ValueError: When the model is pipelined, but no gradient
        accumulation steps have been defined in the gradient accumulation
        settings.
        RuntimeError: When `steps_per_execution` is not divisible by
        `gradient_accumulation_steps`.

    Returns:
        integer: The number of steps to run gradient accumulation over.
    """
    model_mode_message = ""

    if self._is_pipelined():
      model_mode_message = "pipelined "
      if self._pipelining_gradient_accumulation_steps_per_replica is None:
        raise ValueError(
            "The model which you are attempting to train is pipelined, however "
            "`gradient_accumulation_steps_per_replica` has not been set "
            "through `set_pipelining_options()`. You need to set this value as "
            "pipelined models will perform gradient accumulation when "
            "training.")
      gradient_accumulation_steps_per_replica = \
        self._pipelining_gradient_accumulation_steps_per_replica
    else:
      if self._gradient_accumulation_steps_per_replica is None:
        # Non-pipelined models don't need gradient accumulation.
        return 1
      gradient_accumulation_steps_per_replica = \
        self._gradient_accumulation_steps_per_replica

    gradient_accumulation_steps = gradient_accumulation_steps_per_replica

    if steps_per_execution % gradient_accumulation_steps != 0:
      raise RuntimeError(
          "The {}model has been configured to use gradient accumulation for "
          "training, however the current `steps_per_execution` value (set to "
          "{}) is not divisible by `gradient_accumulation_steps_per_replica` "
          "({}). You need to adjust either `steps_per_execution` or"
          "`gradient_accumulation_steps_per_replica` to make sure that "
          "`steps_per_execution` is divisible by "
          "`gradient_accumulation_steps_per_replica`.".format(
              model_mode_message,
              steps_per_execution,
              gradient_accumulation_steps_per_replica,
          ))

    return gradient_accumulation_steps_per_replica

  def _reset_ipu_extension(self):
    """Resets any internal state of the extension when the configuration changes.
    The internal state is represented by:
    - The train function
    - The test function
    - The predict function
    - The replication factor
    - The in- and outfeed managers
    """
    with trackable.no_automatic_dependency_tracking_scope(self):
      self._ipu_train_function = None
      self._ipu_test_function = None
      self._ipu_predict_function = None
      self._replication_factor = None
      self._outfeed_manager = keras_data_feed_manager.OutfeedManager()
      self._infeed_manager = keras_data_feed_manager.InfeedManager()

  def _assert_weights_created_supported(self):
    return True

  def _assert_weights_created_delegate(self):
    if not self.built:
      raise ValueError('Weights for model %s have not yet been created. '
                       'Weights are created when the Model is first called on '
                       'inputs or `build()` is called with an `input_shape`.' %
                       self.name)

  def _reset_compile_cache_supported(self):
    return True

  def _reset_compile_cache_delegate(self):
    self._reset_ipu_extension()
    return self._reset_compile_cache(__extension_delegate=False)

  def _list_functions_for_serialization_supported(self, _):
    return True

  def _list_functions_for_serialization_delegate(self, serialization_cache):
    # SavedModel needs to ignore the execution functions.
    ipu_train_function = self._ipu_train_function
    ipu_test_function = self._ipu_test_function
    ipu_predict_function = self._ipu_predict_function
    self._ipu_train_function = None
    self._ipu_test_function = None
    self._ipu_predict_function = None
    functions = self._list_functions_for_serialization(
        serialization_cache, __extension_delegate=False)
    self._ipu_train_function = ipu_train_function
    self._ipu_test_function = ipu_test_function
    self._ipu_predict_function = ipu_predict_function
    return functions

  def _get_last_batch_results(self, outfeed_queue, replication_factor):
    """Returns the last batch from the outfeed queue (handling replication) if
    synthetic data is not used. Otherwise returns a batch of zeros.

    Args:
        outfeed_queue (IPUOutfeedQueue): The outfeed queue to fetch the data
        from.
        replication_factor (integer): The replication factor.

    Returns:
        Tensor: A Tensor containing only the last batch for every replica.
    """
    if self._use_synthetic_data:
      shapes = outfeed_queue._flat_shapes  # pylint: disable=protected-access
      dtypes = outfeed_queue._flat_types  # pylint: disable=protected-access
      flat_buffers = [
          array_ops.zeros(shape, dtype)
          for shape, dtype in zip(shapes, dtypes)
      ]
      return nest.pack_sequence_as(
          outfeed_queue._structure,  # pylint: disable=protected-access
          flat_buffers)

    results = outfeed_queue.dequeue()
    results = nest.map_structure(lambda x: x[-1], results)

    if replication_factor > 1:
      results = nest.map_structure(lambda x: x[-1], results)

    return results

  def _get_all_batch_results(self, outfeed_queue, replication_factor,
                             num_steps):
    """Returns all the batches of data from the outfeed queue if synthetic data
    is not used. Otherwise returns batches of zeros.

    Args:
        outfeed_queue (IPUOutfeedQueue): The outfeed queue to fetch the data
        from.
        replication_factor (integer): The replication factor.
        num_steps (integer): The number of steps that have been performed to
        complete all batches.

    Returns:
        Tensor: A tensor containing the results of all batches for each replica.
    """
    if self._use_synthetic_data:
      shapes = outfeed_queue._flat_shapes  # pylint: disable=protected-access
      if num_steps > 1:
        shapes = [[shape[0] * num_steps] + shape[1:] for shape in shapes]
      dtypes = outfeed_queue._flat_types  # pylint: disable=protected-access
      flat_buffers = [
          array_ops.zeros(shape, dtype)
          for shape, dtype in zip(shapes, dtypes)
      ]
      return nest.pack_sequence_as(
          outfeed_queue._structure,  # pylint: disable=protected-access
          flat_buffers)

    results = outfeed_queue.dequeue()

    return keras_util.merge_into_batch_dimension(results, replication_factor)

  def _make_single_ipu_train_function(self):
    @def_function.function(experimental_compile=True)
    def train_function(steps_per_execution, iterator, outfeed):
      for _ in math_ops.range(steps_per_execution):
        outfeed.enqueue(self.train_step(next(iterator)))

    return train_function

  def _make_single_ipu_train_function_with_gradient_accumulation(
      self, gradient_accumulation_steps_per_replica):
    # This gradient normalizer can only be enabled when the gradient accumulation
    # reduction method is set to SUM. For other reduction methods, this acts as a
    # pass-through v1 optimizer. Additionally, we do not scale down by
    # replication factor since we use cross_replica_mean for reducing gradients
    # across replicas
    optimizer = _NormalizedKerasOptimizerWrapper(
        self._experimental_gradient_accumulation_normalize_gradients,
        (gradient_accumulation_steps_per_replica), self, self.optimizer)

    optimizer = \
      gradient_accumulation_optimizer.GradientAccumulationOptimizerV2(
          optimizer,
          gradient_accumulation_steps_per_replica,
          reduction_method=self._gradient_accumulation_reduction_method,
          **self._gradient_accumulation_optimizer_kwargs)

    def train_step(data):
      # Implementation of `Model.train_step` with gradient accumulation support.
      data = data_adapter.expand_1d(data)
      x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
      y_pred = self(x, training=True)  # pylint: disable=not-callable
      loss = self.compiled_loss(y,
                                y_pred,
                                sample_weight,
                                regularization_losses=self.losses)
      self.compiled_metrics.update_state(y, y_pred, sample_weight)
      grads_and_vars = optimizer.compute_gradients(loss,
                                                   self.trainable_variables)
      optimizer.apply_gradients(grads_and_vars)
      return {m.name: m.result() for m in self.metrics}

    @def_function.function(experimental_compile=True)
    def train_function(steps_per_execution, iterator, outfeed):
      for _ in math_ops.range(steps_per_execution):
        outfeed.enqueue(train_step(next(iterator)))

    return train_function

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

  def _make_pipeline(self,
                     iterations,
                     gradient_accumulation_steps_per_replica,
                     add_loss=False,
                     add_optimizer=False):
    training = add_loss and add_optimizer
    self._logged_bn_warning = False

    @def_function.function(experimental_compile=True)
    def pipeline_function(_steps_per_execution, iterator, outfeed):
      # Get the shapes for all the inputs.
      input_dtypes = nest.map_structure(lambda spec: spec.dtype,
                                        iterator.element_spec)

      x_dtypes, target_dtypes, sample_weight_dtypes = \
        data_adapter.unpack_x_y_sample_weight(input_dtypes)

      # Get the post order schedule with node to pipeline stage assignment.
      post_order_nodes_and_assignment = self._get_pipeline_post_order()
      last_stage_id = max(post_order_nodes_and_assignment)
      tensor_usage_count = self._tensor_usage_count

      # Dictionaries for mapping processed tensors between stages.
      tensor_dict = OrderedDict()
      num_tensors_per_key = OrderedDict()
      computational_stages = []

      # Targets/sample weights can contain `None`s but they need to be passed
      # around between stages.
      def flatten_without_nones(x):
        output = []
        for t in nest.flatten(x):
          if t is not None:
            output.append(t)
        return output

      def unflatten_and_add_nones(flat_x, structure):
        flat_x_with_nones = []
        next_idx = 0
        for t in nest.flatten(structure):
          if t is None:
            flat_x_with_nones.append(None)
          else:
            flat_x_with_nones.append(flat_x[next_idx])
            next_idx += 1
        return nest.pack_sequence_as(structure, flat_x_with_nones)

      def stage(stage_id, *args, **kwargs):
        if kwargs and not args:
          # When the input from the infeed is a single dict it is treated as
          # kwargs by the pipeline infeed wrapper.
          args = kwargs

        # The index of the first layer to execute - used to skip input layers
        # in stage 0.
        layer_start_idx = 0
        if stage_id == 0:
          # Unpack the data from the infeed.
          data = data_adapter.expand_1d(args)
          inputs, targets, sample_weight = \
            data_adapter.unpack_x_y_sample_weight(data)

          # See functional.call() for details.
          inputs = self._flatten_to_reference_inputs(inputs)
          for input_t in inputs:
            input_t._keras_mask = None  # pylint: disable=protected-access

          for x, y in zip(self.inputs, inputs):
            y = self._conform_to_reference_input(y, ref_input=x)
            x_id = str(id(x))
            tensor_dict[x_id] = [y] * tensor_usage_count[x_id]

          flat_targets = flatten_without_nones(targets)
          flat_sample_weight = flatten_without_nones(sample_weight)
          layer_start_idx = len(self.inputs)

          # Validate inputs
          input_spec.assert_input_compatibility(self.input_spec, inputs,
                                                self.name)
        else:
          tensor_dict.clear()
          start_idx = 0
          # Unpack all the tensors from previous stage.
          for key, num_tensors in num_tensors_per_key.items():
            end_idx = start_idx + num_tensors
            ts = list(args[start_idx:end_idx])
            if key == "targets":
              flat_targets = ts
            elif key == "sample_weights":
              flat_sample_weight = ts
            else:
              tensor_dict[key] = ts
            start_idx = end_idx
        num_tensors_per_key.clear()

        for node in post_order_nodes_and_assignment[stage_id][
            layer_start_idx:]:
          assert not node.is_input

          # Set up the arguments and execute the layer.
          args, kwargs = node.map_arguments(tensor_dict)
          outputs = node.layer(*args, **kwargs)

          # Warn the user of usage of BN layers, as their moving statistics are
          # not properly updated in training pipelines when the momentum is
          # non-trivial.
          if isinstance(node.layer,
                        BatchNormalization) and 0 < node.layer.momentum < 1:
            if not self._logged_bn_warning:
              logging.warn(
                  "The moving statistics for keras.BatchNormalization layers in"
                  " pipelined models are not properly updated when training."
                  " This means that during evaluation or inference, these batch"
                  " normalization layers will not necessarily normalize their"
                  " input data in a way that is consistent with the"
                  " distribution of the data they saw when training.")
              self._logged_bn_warning = True

          # Update tensor_dict.
          for x_id, y in zip(node.flat_output_ids, nest.flatten(outputs)):
            tensor_dict[x_id] = [y] * tensor_usage_count[x_id]

        if stage_id == last_stage_id:
          output_tensors = []
          for x in self.outputs:
            x_id = str(id(x))
            assert x_id in tensor_dict, 'Could not compute output ' + str(x)
            output_tensors.append(tensor_dict[x_id].pop())
          preds = nest.pack_sequence_as(self._nested_outputs, output_tensors)
          targets = unflatten_and_add_nones(flat_targets, target_dtypes)
          sample_weight = unflatten_and_add_nones(flat_sample_weight,
                                                  sample_weight_dtypes)
          if not add_loss:
            return preds

          # Updates stateful loss metrics.
          loss = self.compiled_loss(targets,
                                    preds,
                                    sample_weight,
                                    regularization_losses=self.losses)
          self.compiled_metrics.update_state(targets, preds, sample_weight)
          metrics_output = {m.name: m.result() for m in self.metrics}

          if self._pipelining_accumulate_outfeed:
            for name, val in metrics_output.items():
              metrics_output[
                  name] = val / gradient_accumulation_steps_per_replica

          if add_optimizer:
            return loss, metrics_output
          return metrics_output

        # Pack all the tensors for the next stage.
        all_outputs = []
        for key, tensors in tensor_dict.items():
          num_tensors_per_key[key] = len(tensors)
          all_outputs += list(tensors)
        num_tensors_per_key["targets"] = len(flat_targets)
        all_outputs += flat_targets
        num_tensors_per_key["sample_weights"] = len(flat_sample_weight)
        all_outputs += flat_sample_weight
        return all_outputs

      computational_stages = []
      for stage_id in sorted(post_order_nodes_and_assignment):
        computational_stages.append(partial(stage, stage_id))

      # When training loss and metrics are the outputs from the last
      # computational stage, but we only want to outfeed the metrics so mask out
      # the loss.
      outfeed_mask = [True, False] if training else None

      def optimizer_function(loss, *_):
        # This gradient normalizer can only be enabled when the gradient accumulation
        # reduction method is set to SUM. For other reduction methods, this acts as a
        # pass-through v1 optimizer. Additionally, we do not scale down by
        # replication factor since we use cross_replica_mean for reducing gradients
        # across replicas
        optimizer = _NormalizedKerasOptimizerWrapper(
            self._experimental_pipelining_normalize_gradients,
            (gradient_accumulation_steps_per_replica), self, self.optimizer)
        return pipelining_ops.OptimizerFunctionOutput(optimizer, loss)

      opt = optimizer_function if add_optimizer else None

      accumulate_outfeed = self._pipelining_accumulate_outfeed and (
          add_loss or add_optimizer)

      # Create a dummy call context when evaluating the pipeline op.
      call_context = base_layer_utils.call_context()
      with call_context.enter(layer=self,
                              inputs=[],
                              build_graph=True,
                              training=training):
        pipelining_ops.pipeline(
            computational_stages,
            gradient_accumulation_count=gradient_accumulation_steps_per_replica,
            repeat_count=iterations,
            device_mapping=self._pipelining_device_mapping,
            accumulate_outfeed=accumulate_outfeed,
            inputs=[],
            infeed_queue=iterator._infeed_queue,  # pylint: disable=protected-access
            outfeed_queue=outfeed,
            optimizer_function=opt,
            outfeed_mask=outfeed_mask,
            reduction_method=self._gradient_accumulation_reduction_method,
            **self._pipelining_kwargs)

    return pipeline_function

  def _make_pipeline_ipu_train_function(
      self, iterations, gradient_accumulation_steps_per_replica):
    return self._make_pipeline(iterations,
                               gradient_accumulation_steps_per_replica,
                               add_loss=True,
                               add_optimizer=True)

  def _make_single_ipu_test_function(self):
    @def_function.function(experimental_compile=True)
    def test_function(steps_per_execution, iterator, outfeed):
      for _ in math_ops.range(steps_per_execution):
        outfeed.enqueue(self.test_step(next(iterator)))

    return test_function

  def _make_pipeline_ipu_test_function(self, steps_per_execution):
    return self._make_pipeline(1,
                               steps_per_execution,
                               add_loss=True,
                               add_optimizer=False)

  def _make_single_ipu_predict_function(self):
    @def_function.function(experimental_compile=True)
    def predict_function(steps_per_execution, iterator, outfeed):
      for _ in math_ops.range(steps_per_execution):
        outfeed.enqueue(self.predict_step(next(iterator)))

    return predict_function

  def _make_pipeline_ipu_predict_function(self, steps_per_execution):
    return self._make_pipeline(1,
                               steps_per_execution,
                               add_loss=False,
                               add_optimizer=False)

  def _make_ipu_train_function_wrapper(self):
    def wrapper(pipeline_iterations, gradient_accumulation_steps_per_replica):
      with trackable.no_automatic_dependency_tracking_scope(self):
        need_to_rerun = self._ipu_train_function is None
        if self._is_pipelined():
          # Pipelining needs to embed repeat count and gradient accumulation in
          # the graph.
          need_to_rerun = (
              need_to_rerun or
              self._compiled_pipeline_train_iterations != pipeline_iterations
              or
              self._compiled_pipeline_gradient_accumulation_steps_per_replica
              != gradient_accumulation_steps_per_replica)
        else:
          # Gradient accumulation needs to be embedded in the graph.
          need_to_rerun = (
              need_to_rerun
              or self._compiled_gradient_accumulation_steps_per_replica !=
              gradient_accumulation_steps_per_replica)

        if need_to_rerun:
          if self._make_train_function_overridden():
            raise RuntimeError(
                "The function `make_train_function` for the model {} has been "
                "overridden. This is not supported when using Keras within an "
                "IPUStrategy. Either remove the override from your model "
                "definition or set `enable_keras_extensions=False` when "
                "creating the IPUStrategy.".format(self.name))

          self._validate_call_function()
          if self._is_pipelined():
            if self._train_step_overridden():
              raise RuntimeError(
                  "The function `train_step` for the model {} has been "
                  "overridden. This is not supported for pipelined Keras "
                  "models.".format(self.name))
            self._compiled_pipeline_train_iterations = pipeline_iterations
            self._compiled_pipeline_gradient_accumulation_steps_per_replica = \
              gradient_accumulation_steps_per_replica
            self._ipu_train_function = self._make_pipeline_ipu_train_function(
                pipeline_iterations, gradient_accumulation_steps_per_replica)
          else:
            if gradient_accumulation_steps_per_replica > 1:
              if self._train_step_overridden():
                raise RuntimeError(
                    "The function `train_step` for the model {} has been "
                    "overridden. This is not supported for Keras models with "
                    "gradient accumulation enabled.".format(self.name))
              self._ipu_train_function = \
                self._make_single_ipu_train_function_with_gradient_accumulation(
                    gradient_accumulation_steps_per_replica)
            else:
              self._ipu_train_function = self._make_single_ipu_train_function()
            self._compiled_gradient_accumulation_steps_per_replica = \
              gradient_accumulation_steps_per_replica

      return self._ipu_train_function

    return wrapper

  def _make_ipu_test_function_wrapper(self):
    def wrapper(pipeline_iterations):
      with trackable.no_automatic_dependency_tracking_scope(self):
        need_to_rerun = self._ipu_test_function is None
        if self._is_pipelined():
          # Pipelining needs to embed number of iterations in the graph.
          need_to_rerun = (need_to_rerun
                           or self._compiled_pipeline_test_iterations !=
                           pipeline_iterations)

        if need_to_rerun:
          if self._make_test_function_overridden():
            raise RuntimeError(
                "The function `make_test_function` for the model {} has been "
                "overridden. This is not supported when using Keras within an "
                "IPUStrategy. Either remove the override from your model "
                "definition or set `enable_keras_extensions=False` when "
                "creating the IPUStrategy.".format(self.name))

          self._validate_call_function()
          if self._is_pipelined():
            if self._test_step_overridden():
              raise RuntimeError(
                  "The function `test_step` for the model {} has been "
                  "overridden. This is not supported for pipelined Keras "
                  "models.".format(self.name))
            self._compiled_pipeline_test_iterations = pipeline_iterations
            self._ipu_test_function = self._make_pipeline_ipu_test_function(
                pipeline_iterations)
          else:
            self._ipu_test_function = self._make_single_ipu_test_function()

      return self._ipu_test_function

    return wrapper

  def _make_ipu_predict_function_wrapper(self):
    def wrapper(pipeline_iterations):
      with trackable.no_automatic_dependency_tracking_scope(self):
        need_to_rerun = self._ipu_predict_function is None
        if self._is_pipelined():
          # Pipelining needs to embed number of iterations in the graph.
          need_to_rerun = (need_to_rerun
                           or self._compiled_pipeline_predict_iterations !=
                           pipeline_iterations)

        if need_to_rerun:
          if self._make_predict_function_overridden():
            raise RuntimeError(
                "The function `make_predict_function` for the model {} has "
                "been overridden. This is not supported when using Keras "
                "within an IPUStrategy. Either remove the override from your "
                "model definition or set `enable_keras_extensions=False` when "
                "creating the IPUStrategy.".format(self.name))

          self._validate_call_function()
          if self._is_pipelined():
            if self._predict_step_overridden():
              raise RuntimeError(
                  "The function `predict_step` for the model {} has been "
                  "overridden. This is not supported for pipelined Keras "
                  "models.".format(self.name))
            self._compiled_pipeline_predict_iterations = pipeline_iterations
            self._ipu_predict_function = \
              self._make_pipeline_ipu_predict_function(pipeline_iterations)
          else:
            self._ipu_predict_function = \
              self._make_single_ipu_predict_function()

      return self._ipu_predict_function

    return wrapper

  @trackable.no_automatic_dependency_tracking
  def _set_asynchronous_callbacks_impl(self, asynchronous):
    self._asynchronous_callbacks = asynchronous

  @trackable.no_automatic_dependency_tracking
  def _set_gradient_accumulation_options_impl(
      self, gradient_accumulation_steps_per_replica,
      experimental_normalize_gradients, gradient_accumulation_reduction_method,
      gradient_accumulation_optimizer_kwargs):
    # The extension might need to be reset if any of the values are set.
    reset_extension = False

    if experimental_normalize_gradients and \
        gradient_accumulation_reduction_method != \
          gradient_accumulation_optimizer.GradientAccumulationReductionMethod.SUM: # pylint: disable=line-too-long
      raise ValueError(
          "Setting `experimental_normalize_gradients` to True is only "
          "supported when setting gradient_accumulation_reduction_method "
          "to SUM. It is currently set to "
          f"{gradient_accumulation_reduction_method}")

    self._gradient_accumulation_reduction_method = \
      gradient_accumulation_reduction_method

    if gradient_accumulation_steps_per_replica is not None:
      if not isinstance(gradient_accumulation_steps_per_replica,
                        int) or gradient_accumulation_steps_per_replica < 1:
        raise ValueError(
            "Expected `gradient_accumulation_steps_per_replica` to be a "
            "positive integer, but got {} instead.".format(
                gradient_accumulation_steps_per_replica))
      self._gradient_accumulation_steps_per_replica = \
        gradient_accumulation_steps_per_replica
      reset_extension = True

    if experimental_normalize_gradients is not None:
      self._experimental_gradient_accumulation_normalize_gradients = \
        experimental_normalize_gradients
      reset_extension = True

    if gradient_accumulation_optimizer_kwargs is not None:
      if not isinstance(gradient_accumulation_optimizer_kwargs,
                        (dict, collections_abc.Mapping)):
        raise TypeError(
            "`gradient_accumulation_optimizer_kwargs` must be a dictionary.")

      if "opt" in gradient_accumulation_optimizer_kwargs:
        raise ValueError("Found `opt` key in "
                         "`gradient_accumulation_optimizer_kwargs`. This is "
                         "not supported as the optimizer which the model has "
                         "been compiled with is automatically wrapped.")

      if "num_mini_batches" in gradient_accumulation_optimizer_kwargs:
        raise ValueError("Found `num_mini_batches` key in "
                         "`gradient_accumulation_optimizer_kwargs`. Set the "
                         "`gradient_accumulation_steps_per_replica` argument "
                         "to `set_gradient_accumulation_options` instead.")

      self._gradient_accumulation_optimizer_kwargs = \
        gradient_accumulation_optimizer_kwargs
      reset_extension = True

    if reset_extension:
      self._reset_ipu_extension()

  @trackable.no_automatic_dependency_tracking
  def _set_pipelining_options_impl(
      self, pipelining_gradient_accumulation_steps_per_replica,
      pipelining_device_mapping, accumulate_outfeed,
      experimental_normalize_gradients, gradient_accumulation_reduction_method,
      pipelining_kwargs):
    # The extension might need to be reset if any of the values are set.
    reset_extension = False

    if experimental_normalize_gradients:
      # TODO(T46014) - Change experimental_normalize_gradients to False
      # and set gradient_accumulation_reduction_method to MEAN?
      gradient_accumulation_reduction_method = \
        gradient_accumulation_optimizer.GradientAccumulationReductionMethod.SUM

    self._gradient_accumulation_reduction_method = \
      gradient_accumulation_reduction_method

    if pipelining_gradient_accumulation_steps_per_replica is not None:
      if not isinstance(
          pipelining_gradient_accumulation_steps_per_replica,
          int) or pipelining_gradient_accumulation_steps_per_replica < 1:
        raise ValueError(
            "Expected `gradient_accumulation_steps_per_replica` to be a "
            "positive integer, but got {} instead.".format(
                pipelining_gradient_accumulation_steps_per_replica))
      self._pipelining_gradient_accumulation_steps_per_replica = \
        pipelining_gradient_accumulation_steps_per_replica
      reset_extension = True

    if pipelining_device_mapping is not None:
      if not isinstance(pipelining_device_mapping, list) or not all(
          isinstance(x, int) for x in pipelining_device_mapping):
        raise ValueError("Expected `device_mapping` to be a list of integers.")
      self._pipelining_device_mapping = pipelining_device_mapping
      reset_extension = True

    if accumulate_outfeed is not None:
      self._pipelining_accumulate_outfeed = accumulate_outfeed
      reset_extension = True

    if experimental_normalize_gradients is not None:
      self._experimental_pipelining_normalize_gradients = \
        experimental_normalize_gradients
      reset_extension = True

    if pipelining_kwargs is not None:
      if not isinstance(pipelining_kwargs, (dict, collections_abc.Mapping)):
        raise TypeError("`pipelining_kwargs` must be a dictionary.")

      explicit_args = {
          "gradient_accumulation_count":
          "gradient_accumulation_steps_per_replica",
          "device_mapping": "device_mapping",
          "accumulate_outfeed": "accumulate_outfeed"
      }
      for explicit_arg, alternative in explicit_args.items():
        if explicit_arg in pipelining_kwargs:
          raise ValueError(
              "Found `{}` key in `pipelining_kwargs`. Set the `{}` argument to "
              "`set_pipelining_options` instead.".format(
                  explicit_arg, alternative))

      automatic_args = [
          "computational_stages", "repeat_count", "inputs", "infeed_queue",
          "outfeed_queue", "optimizer_function"
      ]
      for automatic_arg in automatic_args:
        if automatic_arg in pipelining_kwargs:
          raise ValueError(
              "Found `{}` key in `pipelining_kwargs`. This argument is "
              "automatically set by Keras.".format(automatic_arg))

      invalid_args = [
          "outfeed_loss", "outfeed_mask", "batch_serialization_iterations"
      ]
      for invalid_arg in invalid_args:
        if invalid_arg in pipelining_kwargs:
          raise ValueError(
              "Found `{}` key in `pipelining_kwargs`. This argument is "
              "not compatible with Keras.".format(invalid_arg))

      self._pipelining_kwargs = pipelining_kwargs
      reset_extension = True

    if reset_extension:
      self._reset_ipu_extension()

  @trackable.no_automatic_dependency_tracking
  def _set_infeed_queue_options_impl(self, **kwargs):
    automatic_args = ["dataset", "infeed_spec", "element_spec"]
    for automatic_arg in automatic_args:
      if automatic_arg in kwargs:
        raise ValueError("Found `{}` key in `kwargs`. This argument is "
                         "automatically set by Keras.".format(automatic_arg))

    invalid_args = [ipu_infeed_queue._internal_id]  # pylint: disable=protected-access
    for invalid_arg in invalid_args:
      if invalid_arg in kwargs:
        raise ValueError("Found `{}` key in `kwargs`. This argument is "
                         "not compatible with Keras.".format(invalid_arg))

    self._infeed_kwargs = kwargs
    self._reset_ipu_extension()

  @trackable.no_automatic_dependency_tracking
  def _set_outfeed_queue_options_impl(self, **kwargs):
    invalid_args = ["outfeed_mode", "device_ordinal"]
    for invalid_arg in invalid_args:
      if invalid_arg in kwargs:
        raise ValueError("Found `{}` key in `kwargs`. This argument is "
                         "not compatible with Keras.".format(invalid_arg))

    self._outfeed_kwargs = kwargs
    self._reset_ipu_extension()

  def _get_base_config(self):
    """Returns any configuration required to serialize this base class."""
    config = dict()

    config["gradient_accumulation_steps_per_replica"] = \
          self._gradient_accumulation_steps_per_replica
    config["experimental_gradient_accumulation_normalize_gradients"] = \
      self._experimental_gradient_accumulation_normalize_gradients
    config["experimental_pipelining_normalize_gradients"] = \
      self._experimental_pipelining_normalize_gradients
    config["gradient_accumulation_reduction_method"] = \
      self._gradient_accumulation_reduction_method.value
    config["asynchronous_callbacks"] = self._asynchronous_callbacks

    if self._gradient_accumulation_optimizer_kwargs:
      logging.info(
          "Calling get_config() on {} - "
          "`gradient_accumulation_optimizer_kwargs` cannot be serialized and "
          "you will need to call `set_gradient_accumulation_options` again if "
          "the model is restored.".format(self.name))

    config["pipelining_gradient_accumulation_steps_per_replica"] = \
      self._pipelining_gradient_accumulation_steps_per_replica
    config["pipelining_device_mapping"] = self._pipelining_device_mapping
    config["pipelining_accumulate_outfeed"] = \
      self._pipelining_accumulate_outfeed

    config["infeed_kwargs"] = self._infeed_kwargs
    config["outfeed_kwargs"] = self._outfeed_kwargs

    if self._pipelining_kwargs:
      logging.info(
          "Calling get_config() on {} - "
          "`pipelining_kwargs` cannot be serialized and you will need to call "
          "`set_pipelining_options` again if the model is restored.".format(
              self.name))

    return config

  @staticmethod
  def _strip_base_config(config):
    config_new = copy.deepcopy(config)
    base_config_keys = [
        "gradient_accumulation_steps_per_replica",
        "gradient_accumulation_reduction_method",
        "experimental_gradient_accumulation_normalize_gradients",
        "pipelining_gradient_accumulation_steps_per_replica",
        "pipelining_accumulate_outfeed",
        "pipelining_device_mapping",
        "experimental_pipelining_normalize_gradients",
        "asynchronous_callbacks",
        "infeed_kwargs",
        "outfeed_kwargs",
    ]

    for key in base_config_keys:
      try:
        del config_new[key]
      except KeyError:
        pass

    return config_new

  @trackable.no_automatic_dependency_tracking
  def _from_base_config(self, config):
    self._gradient_accumulation_steps_per_replica = config.get(
        "gradient_accumulation_steps_per_replica", None)
    self._experimental_gradient_accumulation_normalize_gradients = config.get(
        "experimental_gradient_accumulation_normalize_gradients", None)
    self._pipelining_gradient_accumulation_steps_per_replica = config.get(
        "pipelining_gradient_accumulation_steps_per_replica", None)
    self._pipelining_device_mapping = config.get("pipelining_device_mapping",
                                                 None)

    reduction_method_int = \
      config.get("gradient_accumulation_reduction_method",
                 gradient_accumulation_optimizer.GradientAccumulationReductionMethod.SUM.value)  # pylint: disable=line-too-long
    self._gradient_accumulation_reduction_method = \
      gradient_accumulation_optimizer.GradientAccumulationReductionMethod(reduction_method_int)  # pylint: disable=line-too-long

    self._pipelining_accumulate_outfeed = config.get(
        "pipelining_accumulate_outfeed", None)
    self._experimental_pipelining_normalize_gradients = config.get(
        "experimental_pipelining_normalize_gradients", None)
    self._asynchronous_callbacks = config.get("asynchronous_callbacks", False)
    self._infeed_kwargs = config.get("infeed_kwargs", dict())
    self._outfeed_kwargs = config.get("outfeed_kwargs", dict())

  def _fit_supported(self, *args, **kwargs):  # pylint:disable=unused-argument
    return True

  def _fit_delegate(self,
                    x=None,
                    y=None,
                    batch_size=None,
                    epochs=1,
                    verbose='auto',
                    callbacks=None,
                    validation_split=0.,
                    validation_data=None,
                    shuffle=True,
                    class_weight=None,
                    sample_weight=None,
                    initial_epoch=0,
                    steps_per_epoch=None,
                    validation_steps=None,
                    validation_batch_size=None,
                    validation_freq=1,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False):
    mode = _Mode.FIT
    base_layer.keras_api_gauge.get_cell('fit').set(True)
    # Legacy graph support is contained in `training_v1.Model`.
    version_utils.disallow_legacy_graph('Model', 'fit')
    self._assert_compile_was_called()
    self._check_call_args('fit')
    training_module._disallow_inside_tf_function('fit')  # pylint: disable=protected-access

    self._check_mode()

    if verbose == 'auto':
      if self.distribute_strategy._should_use_with_coordinator:  # pylint: disable=protected-access
        verbose = 2  # Default to epoch-level logging for PSStrategy.
      else:
        verbose = 1  # Default to batch-level logging otherwise.

    if validation_split:
      # Create the validation data using the training data. Only supported for
      # `Tensor` and `NumPy` input.
      (x, y,
       sample_weight), validation_data = (data_adapter.train_validation_split(
           (x, y, sample_weight), validation_split=validation_split))

    if validation_data:
      val_x, val_y, val_sample_weight = (
          data_adapter.unpack_x_y_sample_weight(validation_data))

    with self.distribute_strategy.scope(), \
         training_utils.RespectCompiledTrainableState(self), \
         libpvti.Tracepoint(_pvti_trace_channel, self.name + ".fit()"):
      # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
      data_handler = ipu_data_adapter.IPUDataHandler(
          x=x,
          y=y,
          sample_weight=sample_weight,
          batch_size=batch_size,
          steps_per_epoch=steps_per_epoch,
          initial_epoch=initial_epoch,
          epochs=epochs,
          shuffle=shuffle,
          class_weight=class_weight,
          max_queue_size=max_queue_size,
          workers=workers,
          use_multiprocessing=use_multiprocessing,
          model=self,
          steps_per_execution=self._steps_per_execution,
          replication_factor=None)

      # Build the model with specific dtypes. This is important for models
      # without explicit input dtypes (model subclasses and some sequential
      # models).
      input_shape, input_dtype = self._get_x_shape_and_dtype(data_handler)
      self._build_with_dtypes(input_shape, input_dtype)

      # Set replication factor after building as we need to know if we are
      # pipelining. Subclassed models don't know if they are pipelining until
      # they have been built.
      replication_factor = self._get_replication_factor()
      data_handler.set_replication_factor(replication_factor)

      self._log_optimizer_batch_size(data_handler)

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        callbacks = callbacks_module.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self,
            verbose=verbose,
            epochs=epochs,
            steps=data_handler.inferred_steps)

      self.stop_training = False
      train_function_wrapper = self._make_ipu_train_function_wrapper()
      self._train_counter.assign(0)
      callbacks.on_train_begin()
      training_logs = None

      data_handler._initial_epoch = (  # pylint: disable=protected-access
          self._maybe_load_initial_epoch_from_ckpt(initial_epoch))
      logs = None

      outfeed = self._outfeed_manager.get_outfeed(mode, self._outfeed_kwargs)  # pylint:disable=unused-variable

      for epoch, iterator in data_handler.enumerate_epochs_with_reuse(
          self._infeed_manager, mode, self._infeed_kwargs):

        inferred_steps = data_handler.inferred_steps
        steps_per_execution = data_handler.steps_per_execution_value

        gradient_accumulation_steps_per_replica = \
          self._verify_and_get_gradient_accumulation_steps_per_replica(
              steps_per_execution)

        # Indicates how many steps the statistics are accumulated over.
        steps_accumulation_factor = (gradient_accumulation_steps_per_replica
                                     if self._pipelining_accumulate_outfeed
                                     else 1)

        # Due to outfeed masking, when pipelined results are wrapped in a list
        # from the outfeed.
        unpack_step_results = self._is_pipelined()

        # Per step callbacks do not make sense if the results are accumulated
        # over multiple steps.
        asynchronous_callbacks = (self._asynchronous_callbacks
                                  and steps_accumulation_factor == 1)

        pipeline_iterations = (steps_per_execution //
                               gradient_accumulation_steps_per_replica)
        train_function = train_function_wrapper(
            pipeline_iterations, gradient_accumulation_steps_per_replica)

        self._log_steps_per_execution_warning(steps_per_execution)

        self.reset_metrics()
        callbacks.on_epoch_begin(epoch)

        def batch_begin_fn(step):
          callbacks.on_train_batch_begin(step)

        def batch_end_fn(step, data):
          training_module.write_scalar_summaries(data, step=step)
          callbacks.on_train_batch_end(step, data)

        outfeed_thread = None
        if asynchronous_callbacks:
          outfeed_thread = keras_polling_thread.PollingThread(
              outfeed,
              inferred_steps,
              replication_factor,
              batch_begin_fn,
              batch_end_fn,
              unpack_step_results=unpack_step_results)
          outfeed_thread.start()

        for step in data_handler.steps():
          end_step = step + data_handler.step_increment
          with trace.Trace('train',
                           epoch_num=epoch,
                           step_num=step,
                           batch_size=batch_size,
                           _r=1):
            if not asynchronous_callbacks:
              batch_begin_fn(step)

            try:
              self.distribute_strategy.run(train_function,
                                           args=(steps_per_execution, iterator,
                                                 outfeed))
            except Exception:  # pylint:disable=broad-except
              if outfeed_thread:
                # Make sure to stop the thread.
                outfeed_thread.cancel()
              six.reraise(*sys.exc_info())

            self._train_counter.assign_add(steps_per_execution)

            if not asynchronous_callbacks:
              data = self._get_last_batch_results(outfeed, replication_factor)
              logs = data[0] if unpack_step_results else data
              batch_end_fn(end_step, logs)

          if self.stop_training:
            if asynchronous_callbacks:
              outfeed_thread.cancel()
            break

        if asynchronous_callbacks:
          outfeed_thread.join()
          logs = outfeed_thread.get_result()

        if logs is None:
          raise ValueError('Expect x to be a non-empty array or dataset.')
        epoch_logs = copy.copy(logs)

        # Run validation.
        if validation_data and self._should_eval(epoch, validation_freq):
          # Create data_handler for evaluation and cache it.
          if getattr(self, '_eval_data_handler', None) is None:
            self._fit_frame = tf_inspect.currentframe()
            self._eval_data_handler = ipu_data_adapter.IPUDataHandler(
                x=val_x,
                y=val_y,
                sample_weight=val_sample_weight,
                batch_size=validation_batch_size or batch_size,
                steps_per_epoch=validation_steps,
                initial_epoch=0,
                epochs=1,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=self,
                steps_per_execution=self._steps_per_execution)
          val_logs = self.evaluate(x=val_x,
                                   y=val_y,
                                   sample_weight=val_sample_weight,
                                   batch_size=validation_batch_size
                                   or batch_size,
                                   steps=validation_steps,
                                   callbacks=callbacks,
                                   max_queue_size=max_queue_size,
                                   workers=workers,
                                   use_multiprocessing=use_multiprocessing,
                                   return_dict=True)
          val_logs = {'val_' + name: val for name, val in val_logs.items()}
          epoch_logs.update(val_logs)

        callbacks.on_epoch_end(epoch, epoch_logs)
        training_logs = epoch_logs
        if self.stop_training:
          break

      # If eval data_hanlder exists, delete it after all epochs are done.
      if getattr(self, '_eval_data_handler', None) is not None:
        del self._eval_data_handler
        del self._fit_frame

      # Delete the outfeed queue.
      with context.eager_mode():
        outfeed.deleter  # pylint: disable=pointless-statement

      callbacks.on_train_end(logs=training_logs)
      return self.history

  def _evaluate_supported(self, *args, **kwargs):  # pylint:disable=unused-argument
    return True

  def _evaluate_delegate(self,
                         x=None,
                         y=None,
                         batch_size=None,
                         verbose=1,
                         sample_weight=None,
                         steps=None,
                         callbacks=None,
                         max_queue_size=10,
                         workers=1,
                         use_multiprocessing=False,
                         return_dict=False,
                         **kwargs):
    mode = _Mode.EVALUATE
    base_layer.keras_api_gauge.get_cell('evaluate').set(True)
    version_utils.disallow_legacy_graph('Model', 'evaluate')
    self._assert_compile_was_called()
    self._check_call_args('evaluate')
    training_module._disallow_inside_tf_function('evaluate')  # pylint: disable=protected-access

    self._check_mode()

    with self.distribute_strategy.scope(), \
         libpvti.Tracepoint(_pvti_trace_channel, self.name + ".evaluate()"):
      # Use cached evaluation data only when it's called in `Model.fit`
      if (getattr(self, '_fit_frame', None) is not None
          and tf_inspect.currentframe().f_back is self._fit_frame
          and getattr(self, '_eval_data_handler', None) is not None):
        data_handler = self._eval_data_handler
      else:
        # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
        data_handler = ipu_data_adapter.IPUDataHandler(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps,
            initial_epoch=0,
            epochs=1,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            model=self,
            steps_per_execution=self._steps_per_execution,
            replication_factor=None)

      # Build the model with specific dtypes. This is important for models
      # without explicit input dtypes (model subclasses and some sequential
      # models).
      input_shape, input_dtype = self._get_x_shape_and_dtype(data_handler)
      self._build_with_dtypes(input_shape, input_dtype)

      # Set replication factor after building as we need to know if we are
      # pipelining. Subclassed models don't know if they are pipelining until
      # they have been built.
      replication_factor = self._get_replication_factor()
      data_handler.set_replication_factor(replication_factor)

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        callbacks = callbacks_module.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self,
            verbose=verbose,
            epochs=1,
            steps=data_handler.inferred_steps)

      logs = {}
      test_function_wrapper = self._make_ipu_test_function_wrapper()

      self._test_counter.assign(0)
      callbacks.on_test_begin()

      outfeed = self._outfeed_manager.get_outfeed(mode, self._outfeed_kwargs)  # pylint:disable=unused-variable

      # Single epoch.
      for _, iterator in data_handler.enumerate_epochs_with_reuse(
          self._infeed_manager, mode, self._infeed_kwargs):
        inferred_steps = data_handler.inferred_steps
        steps_per_execution = data_handler.steps_per_execution_value

        test_function = test_function_wrapper(steps_per_execution)

        self._log_steps_per_execution_warning(steps_per_execution)

        # Indicates how many steps the statistics are accumulated over.
        steps_accumulation_factor = (
            steps_per_execution if self._pipelining_accumulate_outfeed else 1)

        # Due to accumulating, the outfeed is nested.
        unpack_step_results = steps_accumulation_factor > 1

        # Per step callbacks do not make sense if the results are accumulated
        # over multiple steps.
        asynchronous_callbacks = (self._asynchronous_callbacks
                                  and steps_accumulation_factor == 1)

        self.reset_metrics()

        def batch_begin_fn(step):
          callbacks.on_test_batch_begin(step)

        def batch_end_fn(step, data):
          callbacks.on_test_batch_end(step, data)

        outfeed_thread = None
        if asynchronous_callbacks:
          outfeed_thread = keras_polling_thread.PollingThread(
              outfeed,
              inferred_steps,
              replication_factor,
              batch_begin_fn,
              batch_end_fn,
              unpack_step_results=unpack_step_results)
          outfeed_thread.start()

        for step in data_handler.steps():
          end_step = step + data_handler.step_increment
          with trace.Trace('test', step_num=step, _r=1):
            if not asynchronous_callbacks:
              batch_begin_fn(step)

            try:
              self.distribute_strategy.run(test_function,
                                           args=(steps_per_execution, iterator,
                                                 outfeed))
            except Exception:  # pylint:disable=broad-except
              if outfeed_thread:
                # Make sure to stop the thread.
                outfeed_thread.cancel()
              six.reraise(*sys.exc_info())

            self._test_counter.assign_add(steps_per_execution)

            if not asynchronous_callbacks:
              data = self._get_last_batch_results(outfeed, replication_factor)
              logs = data[0] if unpack_step_results else data
              batch_end_fn(end_step, logs)

      if asynchronous_callbacks:
        outfeed_thread.join()
        logs = outfeed_thread.get_result()

      logs = tf_utils.sync_to_numpy_or_python_type(logs)

      # Delete the outfeed queue.
      with context.eager_mode():
        outfeed.deleter  # pylint: disable=pointless-statement

      callbacks.on_test_end(logs=logs)

      if return_dict:
        return logs
      else:
        results = []
        for name in self.metrics_names:
          if name in logs:
            results.append(logs[name])
        for key in sorted(logs.keys()):
          if key not in self.metrics_names:
            results.append(logs[key])
        if len(results) == 1:
          return results[0]
        return results

  def _predict_supported(self, *args, **kwargs):  # pylint:disable=unused-argument
    return True

  def _predict_delegate(self,
                        x,
                        batch_size=None,
                        verbose=0,
                        steps=None,
                        callbacks=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False):
    mode = _Mode.PREDICT
    base_layer.keras_api_gauge.get_cell('predict').set(True)
    version_utils.disallow_legacy_graph('Model', 'predict')
    self._check_call_args('predict')
    training_module._disallow_inside_tf_function('predict')  # pylint: disable=protected-access

    self._check_mode()

    outputs = None
    with self.distribute_strategy.scope(), \
         libpvti.Tracepoint(_pvti_trace_channel, self.name + ".predict()"):
      # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
      data_handler = ipu_data_adapter.IPUDataHandler(
          x=x,
          batch_size=batch_size,
          steps_per_epoch=steps,
          initial_epoch=0,
          epochs=1,
          max_queue_size=max_queue_size,
          workers=workers,
          use_multiprocessing=use_multiprocessing,
          model=self,
          steps_per_execution=self._steps_per_execution,
          replication_factor=None)

      # Build the model with specific dtypes. This is important for models
      # without explicit input dtypes (model subclasses and some sequential
      # models).
      input_shape, input_dtype = self._get_x_shape_and_dtype(data_handler)
      self._build_with_dtypes(input_shape, input_dtype)

      # Set replication factor after building as we need to know if we are
      # pipelining. Subclassed models don't know if they are pipelining until
      # they have been built.
      replication_factor = self._get_replication_factor()
      data_handler.set_replication_factor(replication_factor)

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        callbacks = callbacks_module.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self,
            verbose=verbose,
            epochs=1,
            steps=data_handler.inferred_steps)

      predict_function_wrapper = self._make_ipu_predict_function_wrapper()
      self._predict_counter.assign(0)
      callbacks.on_predict_begin()
      batch_outputs = None

      outfeed = self._outfeed_manager.get_outfeed(mode, self._outfeed_kwargs)  # pylint:disable=unused-variable

      # Single epoch.
      for _, iterator in data_handler.enumerate_epochs_with_reuse(
          self._infeed_manager, mode, self._infeed_kwargs):
        steps_per_execution = data_handler.steps_per_execution_value
        inferred_steps = data_handler.inferred_steps

        predict_function = predict_function_wrapper(steps_per_execution)

        self._log_steps_per_execution_warning(steps_per_execution)

        def batch_begin_fn(step):
          callbacks.on_predict_batch_begin(step)

        def batch_end_fn(step, data):
          callbacks.on_predict_batch_end(step, {'outputs': data})

        def process_batch(outs, batch):
          if outs is None:
            outs = nest.map_structure(lambda batch_output: [batch_output],
                                      batch)
          else:
            nest.map_structure_up_to(
                batch,
                lambda output, batch_output: output.append(batch_output), outs,
                batch)
          return outs

        outfeed_thread = None
        if self._asynchronous_callbacks:
          outfeed_thread = keras_polling_thread.PollingThreadPredict(
              outfeed, inferred_steps, replication_factor, batch_begin_fn,
              batch_end_fn)
          outfeed_thread.start()

        for step in data_handler.steps():
          end_step = step + data_handler.step_increment

          if not self._asynchronous_callbacks:
            batch_begin_fn(step)

          try:
            self.distribute_strategy.run(predict_function,
                                         args=(steps_per_execution, iterator,
                                               outfeed))
          except Exception:  # pylint:disable=broad-except
            if outfeed_thread:
              # Make sure to stop the thread.
              outfeed_thread.cancel()
            six.reraise(*sys.exc_info())

          self._predict_counter.assign_add(steps_per_execution)

          if not self._asynchronous_callbacks:
            batch_outputs = self._get_all_batch_results(
                outfeed, replication_factor, steps_per_execution)
            batch_end_fn(end_step, batch_outputs)
            outputs = process_batch(outputs, batch_outputs)

      if self._asynchronous_callbacks:
        outfeed_thread.join()
        batches = outfeed_thread.get_result()
        for batch in batches:
          batch_outputs = batch
          outputs = process_batch(outputs, batch_outputs)

      if batch_outputs is None:
        raise ValueError('Expect x to be a non-empty array or dataset.')

      # Delete the outfeed queue.
      with context.eager_mode():
        outfeed.deleter  # pylint: disable=pointless-statement

      callbacks.on_predict_end()
    all_outputs = nest.map_structure_up_to(batch_outputs,
                                           training_module.concat, outputs)
    return tf_utils.sync_to_numpy_or_python_type(all_outputs)

  @staticmethod
  def _get_x_shape_and_dtype(data_handler):
    # Extract input shape and dtype from an IPU data handler.
    element_spec = data_handler.element_spec
    x_spec, _, _ = data_adapter.unpack_x_y_sample_weight(element_spec)

    def get_shape(spec):
      # Convert from tensorshapes to tuples of dims.
      shape = spec.shape.as_list()
      if len(shape) == 1:
        # Expand 1d shapes to 2d. This is done automatically to inputs in keras.
        shape.append(1)
      return tuple(shape)

    def get_dtype(spec):
      return spec.dtype

    shapes = nest.map_structure(get_shape, x_spec)
    dtypes = nest.map_structure(get_dtype, x_spec)

    return shapes, dtypes

  def _build_with_dtypes(self, input_shape, input_dtype):
    # Like build, but with the ability to specify input dtypes.
    raise NotImplementedError

  def _get_pipeline_post_order(self):
    """Get a dict of pipeline stage to list of nodes to execute for all the
    nodes in the model. Input layers/nodes are assigned to stage 0."""
    raise NotImplementedError

  def get_pipeline_stage_assignment(self):
    raise NotImplementedError

  def set_pipeline_stage_assignment(self, pipeline_stage_assignment):
    raise NotImplementedError

  def reset_pipeline_stage_assignment(self):
    raise NotImplementedError

  def print_pipeline_stage_assignment_summary(self,
                                              line_length=None,
                                              print_fn=None):
    raise NotImplementedError

  def _print_pipeline_stage_assignment_summary_impl(self, print_assignment_fn,
                                                    headers, column_widths,
                                                    line_length, print_fn):
    """Implementation function for the print_pipeline_stage_assignment_summary.
    """
    print_fn = print_fn if print_fn else print
    assignments = self.get_pipeline_stage_assignment()

    positions = [int(line_length * p) for p in column_widths]

    def print_row(fields):
      assert len(fields) == len(positions)
      line = ''
      for i, field in enumerate(fields):
        if i > 0:
          line = line[:-1] + ' '
        line += str(field)
        line = line[:positions[i]]
        line += ' ' * (positions[i] - len(line))
      print_fn(line)

    print_fn('Model: "{}"'.format(self.name))
    print_fn('_' * line_length)
    print_row(headers)
    print_fn('=' * line_length)

    for i, assignment in enumerate(assignments):
      print_assignment_fn(assignment, print_row)

      if i == len(assignments) - 1:
        print_fn('=' * line_length)
      else:
        print_fn('_' * line_length)

  def _get_pipeline_maximum_pipeline_stage(self):
    """Returns the maximum pipeline stage assignment"""
    raise NotImplementedError

  def _validate_call_function(self):
    """
    Raises an error if call function of the model is incompatible with IPU
    Keras. The requirements vary depending on the type of model.
    """
    raise NotImplementedError

  def _train_step_overridden(self):
    return self.train_step.__func__ != training_module.Model.train_step

  def _make_train_function_overridden(self):
    return (self.make_train_function.__func__ !=
            training_module.Model.make_train_function)

  def _test_step_overridden(self):
    return self.test_step.__func__ != training_module.Model.test_step

  def _make_test_function_overridden(self):
    return (self.make_test_function.__func__ !=
            training_module.Model.make_test_function)

  def _predict_step_overridden(self):
    return self.predict_step.__func__ != training_module.Model.predict_step

  def _make_predict_function_overridden(self):
    return (self.make_predict_function.__func__ !=
            training_module.Model.make_predict_function)
