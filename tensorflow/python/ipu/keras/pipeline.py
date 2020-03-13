# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""
Pipeline interface for Keras
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from functools import partial

from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import def_function
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import Model
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.training.tracking import base as trackable


def next_feed_id(prefix):
  result = prefix + str(next_feed_id.feed_count)
  next_feed_id.feed_count += 1
  return result


next_feed_id.feed_count = 0


class _TensorflowOptimizerWrapper(Optimizer):
  """A class which wraps a standard Tensorflow optimizer, giving it a TF
  optimizer interface, but generating gradients against the Keras Model.
  """
  def __init__(self, model, opt):
    super(_TensorflowOptimizerWrapper, self).__init__(use_locking=False,
                                                      name="optimizer_shim")
    self._model = model
    self._optimizer = opt

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    return self._optimizer.compute_gradients(loss,
                                             self._model.trainable_weights)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    return self._optimizer.apply_gradients(grads_and_vars, global_step, name)

  def _apply_sparse(self, grad, var):
    raise NotImplementedError()

  def _apply_dense(self, grad, var):
    raise NotImplementedError()

  def _resource_apply_dense(self, grad, handle):
    raise NotImplementedError()

  def _resource_apply_sparse(self, grad, handle, indices):
    raise NotImplementedError()


class _KerasOptimizerWrapper(Optimizer):
  """A class which wraps a Keras optimizer, giving it ia TF optimizer interface.
  """
  def __init__(self, model, opt):
    super(_KerasOptimizerWrapper, self).__init__(use_locking=False,
                                                 name="optimizer_shim")
    self._model = model
    self._optimizer = opt

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    grads = self._optimizer.get_gradients(loss, self._model.trainable_weights)
    return zip(grads, self._model.trainable_weights)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    return self._optimizer.apply_gradients(grads_and_vars)

  def _apply_sparse(self, grad, var):
    raise NotImplementedError()

  def _apply_dense(self, grad, var):
    raise NotImplementedError()

  def _resource_apply_dense(self, grad, handle):
    raise NotImplementedError()

  def _resource_apply_sparse(self, grad, handle, indices):
    raise NotImplementedError()


@keras_export(v1=['keras.ipu.PipelinedModel'])
class PipelinedModel(Model):
  """Model for encapsulating a pipeline of stages to be run in parallel on an
  IPU system.
  """
  def __init__(self, stages=None, pipeline_depth=None, **kwargs):
    """Creates a pipelined model.

        Args:
            stages: A python list of lists of Layers.
            pipeline_depth: The number of batches processed by the pipeline on
                            each iteration.
            name: Optional op name.
    """
    name = kwargs.pop("name", None)
    super(PipelinedModel, self).__init__(dtype=None, name=name)

    if not isinstance(stages, list):
      raise ValueError("An IPU pipeline must take a list of stages, where "
                       "each stage is a list of Keras Layers.")

    for s in stages:
      if not isinstance(s, list):
        raise ValueError("An IPU pipeline may only contain lists of "
                         "stages, where each stage is a list of Keras Layers.")

    if not pipeline_depth:
      raise ValueError(
          "The pipeline_depth parameter must be specified.  Choose a "
          "pipeline_depth such that pipeline_depth * mini_batch_size is a "
          "good total batch size.  One step of the model will run "
          "pipeline_depth mini-batches through the model, accumulate the "
          "gradients of the errors, and then apply the accumulated gradients "
          "to the model weights once.")

    self.built = False
    self.pipeline_depth = pipeline_depth
    self.args = kwargs
    self.history = None
    self.stages = stages

  def saveable(self):
    raise NotImplementedError(
        "PipelinedModel does not yet support object-based saving.")

  def build(self, input_shape):
    pass

  # pylint: disable=arguments-differ
  def call(self, _):
    raise ValueError(
        "PipelineModel can only be called through the `fit`, `evaluate` "
        "or `predict` interfaces.")

  @trackable.no_automatic_dependency_tracking
  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              distribute=None,
              **kwargs):

    if isinstance(optimizer, optimizers.Optimizer):
      raise ValueError(
          "Optimizer must be a native Tensorflow optimizers, or Keras V2 "
          "optimizers, found in tensorflow.keras.optimizer_v2.")

    if metrics is not None:
      raise NotImplementedError(
          "metrics currently not supported by the IPU Pipelined model.")

    if not isinstance(loss_weights, (list, type(None))):
      raise ValueError("loss_weights can only be specified as a list.")

    if sample_weight_mode is not None:
      raise NotImplementedError(
          "sample_weight_mode is not supported by the IPU Pipelined model.")

    if weighted_metrics is not None:
      raise NotImplementedError(
          "weighted_metrics is not supported by the IPU Pipelined model.")

    if target_tensors is not None:
      raise NotImplementedError(
          "target_tensors is not supported by the IPU Pipelined model.")

    if distribute is not None:
      raise NotImplementedError(
          "Please compile the model under an IPU distribution scope.")

    if kwargs.pop('run_eagerly', False):
      raise ValueError("PipelineModel cannot be run Eagerly.")

    return super(PipelinedModel,
                 self).compile(optimizer=optimizer,
                               loss=loss,
                               metrics=metrics,
                               loss_weights=loss_weights,
                               sample_weight_mode=sample_weight_mode,
                               weighted_metrics=weighted_metrics,
                               **kwargs)

  # This is a duplication of the graph compilation section of the method
  # Model.compile in training.py
  def _add_loss(self, targets):
    self._is_compiled = True

    target_tensors = targets if isinstance(targets,
                                           (list, tuple)) else [targets]

    # Prepare list of loss functions, same size of model outputs.
    self.loss_functions = training_utils.prepare_loss_functions(
        self.loss, self.output_names)

    self._training_endpoints = []
    for o, n, l, t in zip(self.outputs, self.output_names, self.loss_functions,
                          target_tensors):
      endpoint = training._TrainingEndpoint(o, n, l)  # pylint: disable=protected-access
      endpoint.create_training_target(t, run_eagerly=self.run_eagerly)
      self._training_endpoints.append(endpoint)

    # Prepare list loss weights, same size of model outputs.
    training_utils.prepare_loss_weights(self._training_endpoints,
                                        self.loss_weights)

    masks = self._prepare_output_masks()

    # Save all metric attributes per output of the model.
    self._cache_output_metric_attributes(self._compile_metrics,
                                         self._compile_weighted_metrics)

    # Set metric attributes for each output.
    self._set_metric_attributes()

    # Invoke metric functions (unweighted) for all the outputs.
    self._handle_metrics(self.outputs,
                         targets=self._targets,
                         skip_target_masks=self._prepare_skip_target_masks(),
                         masks=masks)

    # Prepare sample weight modes. List with the same length as model outputs.
    training_utils.prepare_sample_weight_modes(self._training_endpoints, None)

    # Creates the model loss
    self.total_loss = self._prepare_total_loss(masks)

  @def_function.function(autograph=False)
  def _internal_run_loop(self, infeed_queue, outfeed_queue, repeat_count):

    # Plain function to build a stage
    def call_stage(stage_id, inputs, targets):

      # Record the inputs of the first stage
      if stage_id == 0 and not self.inputs:
        self._set_input_attrs(inputs)

      x = inputs
      for l in self.stages[stage_id]:
        x = l(x)

      # Recompile the model now that we know the inputs and outputs, and
      # then create the losses and metrics
      if stage_id == len(self.stages) - 1:
        self._set_output_attrs(x)

        self._add_loss(targets)

        # TODO(T17455) include metric outputs
        return self.total_loss

      return x, targets

    # The pipeline stages, a set of feed forward functions.
    stages = []
    for stage_id in range(len(self.stages)):
      stages.append(partial(call_stage, stage_id))

    # Function for generating the optimizer config for pipelines.
    def optimizer_function(loss):

      if not self.trainable_weights:
        raise ValueError("Model must have at least one trainable parameter.")

      opt = self.optimizer

      # Unwrap native TF optimizers from a Keras wrapper
      if isinstance(opt, optimizers.TFOptimizer):
        opt = _TensorflowOptimizerWrapper(self, opt.optimizer)

      # Convert native Keras optimizers to TF optimizers
      elif isinstance(opt, optimizer_v2.OptimizerV2):
        opt = _KerasOptimizerWrapper(self, opt)

      # Other optimizer types are not supported
      else:
        raise ValueError(
            "Only Keras optimizer_v2.Optimizer and Tensorflow native "
            "training.Optimizer subclasses are supported.")

      return pipelining_ops.OptimizerFunctionOutput(opt, loss)

    pipeline = pipelining_ops.pipeline(stages,
                                       pipeline_depth=self.pipeline_depth,
                                       repeat_count=repeat_count,
                                       inputs=[],
                                       infeed_queue=infeed_queue,
                                       outfeed_queue=outfeed_queue,
                                       optimizer_function=optimizer_function,
                                       name=self.name,
                                       **self.args)

    return pipeline.outputs

  @def_function.function(autograph=False)
  def _run_helper(self, infeed_queue, outfeed_queue=None, repeat_count=1):

    strategy = distribution_strategy_context.get_strategy()
    return strategy.experimental_run_v2(
        self._internal_run_loop,
        args=[infeed_queue, outfeed_queue, repeat_count])

  @trackable.no_automatic_dependency_tracking
  def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False,
          steps_per_run=None,
          **kwargs):

    if not isinstance(x, dataset_ops.DatasetV2):
      raise ValueError(
          "PipelineModels can only `fit` with a `tf.data.Dataset` as input.")

    if y is not None:
      raise ValueError(
          "When `x` is a `tf.data.Dataset`, `y` should not be specified "
          "(since targets will be obtained from `x`).")

    if validation_split != 0. or validation_steps != None:
      raise ValueError("PipelineModels do not support validation during fit.")

    self._assert_compile_was_called()

    mode = ModeKeys.TRAIN

    # Figure out if we need to recreate the iterator after each epoch.
    recreate_iterator = False
    require_steps_per_epoch = False
    verify_dataset_length = False

    dataset_length = backend.get_value(cardinality.cardinality(x))
    if dataset_length == cardinality.INFINITE:
      # An infinite dataset can be walked over indefinitely, but the user
      # must specify how many steps there are in each epoch.
      require_steps_per_epoch = True
    elif dataset_length == cardinality.UNKNOWN:
      # An unknown length of dataset must be restarted on each epoch, and
      # the user must specify the number of steps per epoch.
      require_steps_per_epoch = True
      recreate_iterator = True
    else:
      # A known length of dataset must be restarted on each epoch, but the
      # user doesn't have to specify the number of steps per epoch.
      recreate_iterator = True
      verify_dataset_length = True

    if require_steps_per_epoch:
      if not steps_per_epoch:
        raise ValueError(
            "When using an infinitely repeating dataset, you must provide "
            "the number of steps per epoch (steps_per_epoch).")

    # Find out how many mini-batches, steps, pipeline repeats, and outer loops.
    mini_batches_per_epoch = steps_per_epoch
    if mini_batches_per_epoch is not None:
      mini_batches_per_epoch = mini_batches_per_epoch * self.pipeline_depth

    # If there is a fixed length of dataset, and the user has also specified
    # a steps_per_epoch, then check that this won't exhaust the dataset.
    if verify_dataset_length and steps_per_epoch:
      if mini_batches_per_epoch > dataset_length:
        raise ValueError(
            "Steps per epoch times pipeline depth (%d x %d) is greater than "
            "the number of samples in the dataset (%d)." %
            (steps_per_epoch, self.pipeline_depth, dataset_length))

    mini_batches_per_epoch = training_utils.infer_steps_for_dataset(
        self, x, mini_batches_per_epoch, epochs, steps_name='steps_per_epoch')

    if mini_batches_per_epoch % self.pipeline_depth != 0:
      raise ValueError(
          "PipelineModels require the number of batches in the dataset (%d) "
          "to be a multiple of the pipeline depth (%d)" %
          (mini_batches_per_epoch, self.pipeline_depth))

    steps_per_epoch = mini_batches_per_epoch / self.pipeline_depth

    if not steps_per_run:
      steps_per_run = steps_per_epoch

    if steps_per_epoch % steps_per_run != 0:
      raise ValueError(
          "PipelineModels require the number of steps per execution of the "
          "on device training loop 'steps_per_run' (%d) to be a multiple of "
          "the number of steps in the epoch (%d)." %
          (mini_batches_per_epoch, self.pipeline_depth))

    outer_loop_count = int(steps_per_epoch / steps_per_run)

    # Prepare for progress reporting
    callbacks = cbks.configure_callbacks(callbacks,
                                         self,
                                         epochs=epochs,
                                         steps_per_epoch=steps_per_epoch,
                                         verbose=verbose,
                                         count_mode='steps',
                                         mode=mode)

    # Create queues
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(x, next_feed_id("infeed"))

    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id("outfeed"))

    initial_epoch = self._maybe_load_initial_epoch_from_ckpt(
        initial_epoch, mode)

    callbacks.on_train_begin(mode)

    # Ask the poplar executor to create a dataset iterator
    infeed_queue.initializer  # pylint: disable=pointless-statement

    # Training
    for epoch in range(initial_epoch, epochs):
      if callbacks.model.stop_training:
        break

      epoch_logs = {}
      callbacks.on_epoch_begin(epoch, epoch_logs)

      # Clear metrics
      self.reset_metrics()

      for run in range(outer_loop_count):

        batch_num = run * steps_per_run
        batch_logs = {
            'batch': batch_num,
            'size': 1,
            'num_steps': steps_per_run
        }
        callbacks.on_batch_begin(batch_num, batch_logs)

        # Create and run the pipeline.
        self._run_helper(infeed_queue,
                         outfeed_queue,
                         repeat_count=steps_per_run)

        # Send an end of batches
        callbacks.on_batch_end(batch_num, batch_logs)

      # Restart the iterator at the end of the epoch if necessary
      if recreate_iterator:
        infeed_queue.deleter  # pylint: disable=pointless-statement
        infeed_queue.initializer  # pylint: disable=pointless-statement

      # Fetch the outfeed for the history
      results = outfeed_queue.dequeue()

      # Get the final loss and metrics
      results = map(lambda x: x.numpy(), results)
      results = list(map(lambda x: list(x)[-1], results))

      assert len(self.metrics_names) == len(results)

      # Store only the final losses/metrics for the epoch log
      cbks.make_logs(self, epoch_logs, results, mode)
      callbacks.on_epoch_end(epoch, epoch_logs)

    callbacks.on_train_end(mode)

    # Close the infeed queue at the end of the epoch
    infeed_queue.deleter  # pylint: disable=pointless-statement

    # Close the outfeed queue iterator.
    outfeed_queue.deleter  # pylint: disable=pointless-statement

    return self.history

  def evaluate(self,
               x=None,
               y=None,
               batch_size=None,
               verbose=1,
               sample_weight=None,
               steps=None,
               callbacks=None,
               max_queue_size=10,
               workers=1,
               use_multiprocessing=False):
    raise NotImplementedError(
        "PipelineModels does not support the `evaluate` interface.")

  def predict(self,
              x,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False):
    raise NotImplementedError(
        "PipelineModels does not support the `predict` interface.")
