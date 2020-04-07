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
# ===================================================================
"""
IPUEstimator
~~~~~~~~~~~~
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import itertools
import threading

from six.moves import _thread
import six

from tensorflow.compiler.plugin.poplar.driver.config_pb2 import IpuOptions
from tensorflow.compiler.plugin.poplar.ops import gen_pop_datastream_ops
from tensorflow.compiler.plugin.poplar.ops import gen_sendrecv_ops
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ipu import autoshard
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import ipu_run_config
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import ops as ipu_ops
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu.ipu_multi_worker_strategy import IPUMultiWorkerStrategy
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.util import function_utils

_INITIAL_LOSS = 0.0
_INPUT_FN_KEY = "input_fn"
_ASSIGN_ADD_OP = "AssignAddVariableOp"
_CROSS_REPLICA_SUM_OP = "IpuCrossReplicaSum"
_HOST_DEVICE = "/device:CPU:0"
_IPU_DEVICE = "/device:IPU:0"

# Keys that cannot be used in the `params` dictionary passed to the
# IPUEstimator
_RESERVED_PARAMS_KEYS = [_INPUT_FN_KEY]


class IPUEstimatorSpec(
    collections.namedtuple('IPUEstimatorSpec', [
        'mode', 'predictions', 'loss', 'train_op', 'eval_metric_ops',
        'host_call', 'training_hooks', 'evaluation_hooks', 'prediction_hooks'
    ])):
  """Ops and objects returned from a `model_fn` and passed to `IPUEstimator`."""
  def __new__(cls,
              mode,
              predictions=None,
              loss=None,
              train_op=None,
              eval_metric_ops=None,
              host_call=None,
              training_hooks=None,
              evaluation_hooks=None,
              prediction_hooks=None):
    train_op = model_fn_lib._validate_estimator_spec_train_op(train_op, mode)
    loss = model_fn_lib._validate_estimator_spec_loss(loss, mode)
    predictions = model_fn_lib._validate_estimator_spec_predictions(
        predictions, mode)
    training_hooks = model_fn_lib._validate_estimator_spec_hooks(
        training_hooks)
    evaluation_hooks = model_fn_lib._validate_estimator_spec_hooks(
        evaluation_hooks)
    prediction_hooks = model_fn_lib._validate_estimator_spec_hooks(
        prediction_hooks)
    eval_metric_ops = model_fn_lib._validate_eval_metric_ops(eval_metric_ops)

    if host_call is not None:
      if not isinstance(host_call, tuple):
        raise ValueError("`host_call` must be a tuple")
      if len(host_call) != 2:
        raise ValueError("`host_call` must have two elements")
      if not isinstance(host_call[1], list):
        raise ValueError("second element in `host_call` must be a list")

    return super().__new__(cls,
                           mode=mode,
                           predictions=predictions,
                           loss=loss,
                           train_op=train_op,
                           eval_metric_ops=eval_metric_ops,
                           host_call=host_call,
                           training_hooks=training_hooks,
                           evaluation_hooks=evaluation_hooks,
                           prediction_hooks=prediction_hooks)


class _IPUConfigureIPUSystemHook(session_run_hook.SessionRunHook):
  def __init__(self, config, host_device=_HOST_DEVICE):
    if not isinstance(config, IpuOptions):
      raise Exception("`config` must be an IpuOptions instance")
    self._config = config
    self._host_device = host_device

  def begin(self):
    ipu_utils.configure_ipu_system(self._config, self._host_device)


class _IPUInfeedLifecycleHook(session_run_hook.SessionRunHook):
  def __init__(self, infeed):
    self._infeed = infeed
    self._should_delete = False

  def after_create_session(self, session, coord):
    session.run(self._infeed.initializer)
    self._should_delete = True

  def end(self, session):
    session.run(self._infeed.deleter)
    self._should_delete = False

  def _run_delete_op_in_new_graph_and_session(self):
    g = ops.Graph()
    with g.as_default(), ops.device(_HOST_DEVICE):
      delete_op = gen_pop_datastream_ops.ipu_delete_dataset_iterator(
          feed_id=self._infeed._id)  # pylint: disable=protected-access
    with session_lib.Session(graph=g) as sess:
      sess.run(delete_op)

  def __del__(self):
    if self._should_delete:
      # We may end up here if the session exited abnormally, such
      # as if an exception was raised or the generator returned
      # by `predict()` was deleted, since these scenarios will
      # not trigger the `end()` callback above.
      self._run_delete_op_in_new_graph_and_session()


class _IPUOutfeedLifecycleHook(session_run_hook.SessionRunHook):
  def __init__(self, outfeed):
    self._outfeed = outfeed
    self._should_delete = False

  def after_run(self, run_context, run_values):
    # The outfeed is allocated when the engine is executed.
    self._should_delete = True

  def _run_delete_op_in_new_graph_and_session(self):
    g = ops.Graph()
    with g.as_default(), ops.device(_HOST_DEVICE):
      delete_op = gen_pop_datastream_ops.ipu_delete_outfeed(
          feed_id=self._outfeed._feed_name)  # pylint: disable=protected-access
    with session_lib.Session(graph=g) as sess:
      sess.run(delete_op)

  def __del__(self):
    if self._should_delete:
      self._run_delete_op_in_new_graph_and_session()


class _FeedIdAllocator:
  """Allocates feed IDs with maximum reuse to minimize recompilations.
  The feeds are deleted after each IPUEstimator function call,
  so the IDs only need to be different for function calls that can
  overlap (e.g. from different threads)."""

  _lock = threading.Lock()  # Protecting all class members.
  _thread_ids = []  # All the threads that have ever allocated an ID.

  @classmethod
  def _alloc_thread_index(cls):
    thread_id = _thread.get_ident()

    with cls._lock:
      if not thread_id in cls._thread_ids:
        cls._thread_ids.append(thread_id)
      index = cls._thread_ids.index(thread_id)

    assert index >= 0
    return index

  @classmethod
  def alloc_infeed_id(cls, mode):
    index = cls._alloc_thread_index()
    return "ipu_estimator_{}_{}_infeed".format(index, mode)

  @classmethod
  def alloc_outfeed_id(cls, mode):
    index = cls._alloc_thread_index()
    return "ipu_estimator_{}_{}_outfeed".format(index, mode)


class _IPUGlobalStepCounterAndStopHook(session_run_hook.SessionRunHook):
  def __init__(self, iterations_per_loop, num_steps, final_step):
    if num_steps is None and final_step is None:
      raise ValueError("One of `num_steps` or `final_step` must be specified.")
    if num_steps is not None and final_step is not None:
      raise ValueError(
          "Only one of `num_steps` or `final_step` can be specified.")

    self._iterations_per_loop = iterations_per_loop
    self._num_steps = num_steps
    self._final_step = final_step

  def after_create_session(self, session, coord):
    global_step = session.run(self._global_step_tensor)
    if self._final_step is None:
      self._final_step = global_step + self._num_steps

  def begin(self):
    self._global_step_tensor = training_util.get_global_step()
    with ops.device(_HOST_DEVICE):
      self._increment_op = self._global_step_tensor.assign_add(
          self._iterations_per_loop)

  def after_run(self, run_context, run_values):
    global_step = run_context.session.run(self._increment_op)
    if global_step >= self._final_step:
      run_context.request_stop()


def _call_input_fn(input_fn, mode, params, config, input_context):
  input_fn_args = function_utils.fn_args(input_fn)
  kwargs = {}
  if "mode" in input_fn_args:
    kwargs["mode"] = mode
  if "params" in input_fn_args:
    kwargs["params"] = params
  if "config" in input_fn_args:
    kwargs["config"] = config
  if input_context and "input_context" in input_fn_args:
    kwargs["input_context"] = input_context
  with ops.device(_HOST_DEVICE):
    return input_fn(**kwargs)


def _validate_global_step_not_incremented():
  operations = ops.get_default_graph().get_operations()
  for op in operations:
    if op.type == _ASSIGN_ADD_OP and "global_step" in op.inputs[0].name:
      raise ValueError(
          "Illegal increment of the `global_step` variable in the `model_fn`. "
          "This is usually caused by passing it as an argument to the "
          "`Optimizer.minimize()` function. Please remove this argument as "
          "the IPUEstimator itself is responsible for incrementing it.")


def _validate_replicated_training_graph():
  operations = ops.get_default_graph().get_operations()
  if not any(op.type == _CROSS_REPLICA_SUM_OP for op in operations):
    raise ValueError(
        ("This is not a valid replicated training graph because no {} " +
         "operations were found. Did you remember to use the " +
         "`tensorflow.python.ipu.optimizers.`" +
         "`cross_replica_optimizer.CrossReplicaOptimizer`?"
         ).format(_CROSS_REPLICA_SUM_OP))


def _add_send_to_host_ops(tensors):
  """Returns attributes for matching recv ops"""
  recv_ops_attrs = []

  for tensor in tensors:
    model_fn_lib._check_is_tensor_or_operation(  # pylint: disable=protected-access
        tensor, "`host_call` argument")

    attrs = dict(tensor_name=tensor.name,
                 send_device=_IPU_DEVICE,
                 send_device_incarnation=0,
                 recv_device=_HOST_DEVICE)

    gen_sendrecv_ops.ipu_send_to_host(tensor, **attrs)

    # The recv op has an additional type argument.
    attrs["T"] = tensor.dtype
    recv_ops_attrs.append(attrs)

  return recv_ops_attrs


def _add_recv_at_host_ops(recv_ops_attrs):
  tensors = []
  for attrs in recv_ops_attrs:
    tensors.append(gen_sendrecv_ops.ipu_recv_at_host(**attrs))
  return tensors


def _unpack_features_and_labels(args, kwargs):
  if args and kwargs:
    raise ValueError("Invalid dataset with both tuple and keywords")
  if not args and not kwargs:
    raise ValueError("Invalid dataset with neither tuple nor keywords")

  if args:
    if len(args) == 1:
      features = args[0]
      labels = None
    elif len(args) == 2:
      features, labels = args
    else:
      raise ValueError(
          "Invalid dataset tuple, expected 1 or 2 elements, got {}".format(
              len(args)))
  else:
    features = kwargs
    labels = None

  return features, labels


def _extract_metric_values(eval_dict):
  metric_values = {}

  # Sort metrics lexicographically so graph is identical every time.
  for name, (value_tensor, update_op) in sorted(six.iteritems(eval_dict)):
    # We cannot depend on the `value_tensor` as it is unspecified whether it
    # is evaluated before or after the `update_op`. For example there are no
    # control dependencies between the `assign_add()` update ops and the tensor
    # in `metrics_impl.mean()`. There does however seem to be a guarantee that
    # the `update_op` returns the updated value, so we will just ignore the
    # `value_tensor` and use the result of the `update_op` instead.
    del value_tensor
    model_fn_lib._check_is_tensor(update_op, "update_op")  # pylint: disable=protected-access
    metric_values[name] = update_op

  return metric_values


@six.add_metaclass(abc.ABCMeta)
class _ModelFnWrapperBase:
  """Interface for wrapping the user-provided `model_fn` in a loop."""
  @abc.abstractproperty
  def captured_hooks(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def create_training_loop(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_training_loss_and_op(self, compiled_training_loop):
    raise NotImplementedError()

  @abc.abstractmethod
  def create_evaluation_loop(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_evaluation_loss_and_metrics(self, compiled_evaluation_loop):
    raise NotImplementedError()

  @abc.abstractmethod
  def create_prediction_loop(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_predictions(self, compiled_prediction_loop):
    raise NotImplementedError()

  @staticmethod
  @abc.abstractmethod
  def get_outfeed_mode(mode):
    raise NotImplementedError()


class _ModelFnWrapper(_ModelFnWrapperBase):
  def __init__(self, model_fn, config, params, infeed_queue, outfeed_queue):
    self._model_fn = model_fn
    self._config = config
    self._params = params
    self._infeed_queue = infeed_queue
    self._outfeed_queue = outfeed_queue
    self._iterations_per_loop = config.ipu_run_config.iterations_per_loop
    self._replication_factor = config.ipu_run_config.num_replicas
    self._num_shards = config.ipu_run_config.num_shards
    self._autosharding = config.ipu_run_config.autosharding
    self._captured_hooks = []
    self._captured_host_call_fn = None
    self._captured_host_call_args = None

  @staticmethod
  def get_outfeed_mode(mode):
    if mode == model_fn_lib.ModeKeys.PREDICT:
      return ipu_outfeed_queue.IPUOutfeedMode.ALL
    if mode == model_fn_lib.ModeKeys.EVAL:
      return ipu_outfeed_queue.IPUOutfeedMode.LAST
    # No outfeed for training
    return None

  def _loop_replica_mean(self, loop_sum):
    if self._replication_factor == 1:
      return loop_sum / self._iterations_per_loop

    loop_replica_sum = ipu_ops.cross_replica_ops.cross_replica_sum(loop_sum)
    return loop_replica_sum / (self._iterations_per_loop *
                               self._replication_factor)

  def _capture_hooks(self, hooks):
    if hooks:
      assert not self._captured_hooks, "Can only capture hooks once"
      self._captured_hooks = hooks

  @property
  def captured_hooks(self):
    return self._captured_hooks

  def _capture_host_call(self, host_call):
    if host_call:
      assert self._captured_host_call_fn is None, "Can only capture host_call once"
      self._captured_host_call_fn, tensors = host_call
      self._captured_host_call_args = _add_send_to_host_ops(tensors)

  def _received_host_call_args(self):
    return _add_recv_at_host_ops(self._captured_host_call_args)

  def create_training_loop(self):
    def training_step(total_loss, *args, **kwargs):
      features, labels = _unpack_features_and_labels(args, kwargs)
      estimator_spec = self._call_model_fn(features, labels,
                                           model_fn_lib.ModeKeys.TRAIN)

      loss = estimator_spec.loss
      if loss is None:
        raise ValueError("EstimatorSpec must contain loss when training")

      train_op = estimator_spec.train_op
      if train_op is None:
        raise ValueError("EstimatorSpec must contain train_op when training")

      self._capture_hooks(estimator_spec.training_hooks)

      if isinstance(estimator_spec, IPUEstimatorSpec):
        self._capture_host_call(estimator_spec.host_call)

      if self._autosharding:
        autoshard.automatic_sharding(self._num_shards, features, loss)

      # training_step will be run by xla.compile(). xla.compile() only supports
      # tensor output while train_op can be either an operation or a tensor.
      # Even though xla.compile() automatically adds operation-typed train_op as
      # control dependency of other tensor outputs, it doesn't do so for
      # tensor-typed train_op. Thus, we need to set it explicitly here.
      with ops.control_dependencies([train_op]):
        total_loss += math_ops.cast(loss, dtypes.float32)

      _validate_global_step_not_incremented()

      if self._replication_factor > 1:
        _validate_replicated_training_graph()

      return total_loss

    def training_loop():
      if self._iterations_per_loop == 1:
        # Simplify the graph by avoiding the loop.
        inputs = self._infeed_queue._dequeue()  # pylint: disable=protected-access
        args, kwargs = loops._body_arguments(inputs)  # pylint: disable=protected-access
        total_loss = training_step(_INITIAL_LOSS, *args, **kwargs)
        return total_loss

      total_loss = loops.repeat(self._iterations_per_loop,
                                training_step,
                                inputs=[_INITIAL_LOSS],
                                infeed_queue=self._infeed_queue)

      if self._captured_host_call_fn is not None:
        raise ValueError(
            "host_call is not allowed for iterations_per_loop > 1")

      return self._loop_replica_mean(total_loss)

    return training_loop

  def get_training_loss_and_op(self, compiled_training_loop):
    loss = compiled_training_loop[0]

    if self._captured_host_call_fn is None:
      train_op = loss
    else:
      # The base class will run both `train_op` and `loss`.
      # Let `train_op` be the return value from the host call.
      # If there is a dependency on the `loss` calculated on
      # the IPU, they will be sequenced. Otherwise they might
      # run in parallel on the IPU and CPU.
      with ops.device(_HOST_DEVICE):
        train_op = _call_host_fn(self._captured_host_call_fn,
                                 self._received_host_call_args())

    return loss, train_op

  def create_evaluation_loop(self):
    def evaluation_step(total_loss, *args, **kwargs):
      features, labels = _unpack_features_and_labels(args, kwargs)
      estimator_spec = self._call_model_fn(features, labels,
                                           model_fn_lib.ModeKeys.EVAL)

      loss = estimator_spec.loss
      if loss is None:
        raise ValueError("EstimatorSpec must contain loss when evaluating")

      eval_metric_ops = estimator_spec.eval_metric_ops
      if not eval_metric_ops:
        raise ValueError(
            "EstimatorSpec must contain eval_metric_ops when evaluating")

      self._capture_hooks(estimator_spec.evaluation_hooks)

      metric_values = _extract_metric_values(eval_metric_ops)

      if self._autosharding:
        autoshard.automatic_sharding(self._num_shards, features, loss)

      total_loss += math_ops.cast(loss, dtypes.float32)
      outfeed = self._outfeed_queue.enqueue(metric_values)
      return total_loss, outfeed

    def evaluation_loop():
      total_loss = loops.repeat(self._iterations_per_loop,
                                evaluation_step,
                                inputs=[_INITIAL_LOSS],
                                infeed_queue=self._infeed_queue)
      return self._loop_replica_mean(total_loss)

    return evaluation_loop

  def get_evaluation_loss_and_metrics(self, compiled_evaluation_loop):
    loss = compiled_evaluation_loop[0]

    with ops.device(_HOST_DEVICE):
      metrics = self._outfeed_queue.dequeue()
      metric_ops = {}
      for metric_name, metric_tensor in six.iteritems(metrics):
        # For replicated graphs the tensor will have an additional replica
        # dimension, so we reduce over this dimension (if it exists).
        # TODO(hakons): mean is not always correct, e.g. for
        # root_mean_squared_error.
        # Use no-op as the update_op since updating is done inside the loop.
        metric_ops[metric_name] = (math_ops.reduce_mean(metric_tensor),
                                   control_flow_ops.no_op())

    return loss, metric_ops

  def create_prediction_loop(self):
    def prediction_step(*args, **kwargs):
      features, _ = _unpack_features_and_labels(args, kwargs)
      labels = None  # Do not provide labels for prediction
      estimator_spec = self._call_model_fn(features, labels,
                                           model_fn_lib.ModeKeys.PREDICT)

      predictions = estimator_spec.predictions
      if predictions is None:
        raise ValueError(
            "EstimatorSpec must contain predictions when predicting")

      self._capture_hooks(estimator_spec.prediction_hooks)

      outfeed = self._outfeed_queue.enqueue(predictions)
      return outfeed

    def prediction_loop():
      return loops.repeat(self._iterations_per_loop,
                          prediction_step,
                          infeed_queue=self._infeed_queue)

    return prediction_loop

  def get_predictions(self, compiled_prediction_loop):
    with ops.device(_HOST_DEVICE):
      with ops.control_dependencies([compiled_prediction_loop]):
        predictions = self._outfeed_queue.dequeue()
    return predictions

  def _call_model_fn(self, features, labels, mode):
    model_fn_args = function_utils.fn_args(self._model_fn)
    kwargs = {}
    if "labels" in model_fn_args:
      kwargs["labels"] = labels
    else:
      if labels is not None:
        raise ValueError(
            "model_fn does not take labels, but input_fn returns labels.")
    if "mode" in model_fn_args:
      kwargs["mode"] = mode
    if "params" in model_fn_args:
      kwargs["params"] = self._params
    if "config" in model_fn_args:
      kwargs["config"] = self._config

    estimator_spec = self._model_fn(features=features, **kwargs)

    valid_classes = (IPUEstimatorSpec, model_fn_lib.EstimatorSpec)
    if not isinstance(estimator_spec, valid_classes):
      raise ValueError("`model_fn` must return {}".format(" or ".join(
          [cls.__name__ for cls in valid_classes])))

    return estimator_spec


def _call_host_fn(host_call_fn, host_call_args):
  assert host_call_fn is not None
  assert host_call_args is not None

  ret = host_call_fn(*host_call_args)

  model_fn_lib._check_is_tensor_or_operation(  # pylint: disable=protected-access
      ret, "`host_call` return value")

  return ret


def _get_input_context():
  strategy = distribution_strategy_context.get_strategy()
  if isinstance(strategy, IPUMultiWorkerStrategy):
    return strategy.extended._make_input_context()  # pylint: disable=protected-access
  return None


def _augment_model_fn(model_fn, wrapper_class):
  """Wraps the `model_fn`, feeds it with queues, and returns a new
  `model_fn` that returns a regular `EstimatorSpec`. This `model_fn` wraps
  all the IPU support and can be passed to the regular `Estimator` class."""
  def _model_fn(features, labels, mode, config, params):
    del features, labels  # We call the input_fn directly from here instead
    input_fn = params[_INPUT_FN_KEY]
    input_context = _get_input_context()
    dataset = _call_input_fn(input_fn, mode, params, config, input_context)

    # DatasetV1 (the current alias of Dataset) inherits
    # from DatasetV2, so this allows both.
    if not isinstance(dataset, dataset_ops.DatasetV2):
      raise ValueError("input_fn must return Dataset")

    hooks = []
    replication_factor = config.ipu_run_config.num_replicas

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset,
        _FeedIdAllocator.alloc_infeed_id(mode),
        replication_factor=replication_factor)
    hooks.append(_IPUInfeedLifecycleHook(infeed_queue))

    outfeed_mode = wrapper_class.get_outfeed_mode(mode)
    if outfeed_mode is None:
      outfeed_queue = None
    else:
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(
          _FeedIdAllocator.alloc_outfeed_id(mode),
          outfeed_mode=outfeed_mode,
          replication_factor=replication_factor)
      hooks.append(_IPUOutfeedLifecycleHook(outfeed_queue))

    if config.ipu_run_config.ipu_options is None:
      if config.ipu_run_config.compile_summary:
        logging.warning(
            "Compile summary enabled but IpuOptions is None. No profile will be generated"
        )

    if config.ipu_run_config.ipu_options is not None:
      ipu_options = config.ipu_run_config.ipu_options
      hooks.append(
          _IPUConfigureIPUSystemHook(ipu_options, host_device=_HOST_DEVICE))

    wrapped_model_fn = wrapper_class(model_fn, config, params, infeed_queue,
                                     outfeed_queue)

    if mode == model_fn_lib.ModeKeys.TRAIN:
      loop = wrapped_model_fn.create_training_loop()
    elif mode == model_fn_lib.ModeKeys.EVAL:
      loop = wrapped_model_fn.create_evaluation_loop()
    elif mode == model_fn_lib.ModeKeys.PREDICT:
      loop = wrapped_model_fn.create_prediction_loop()
    else:
      raise ValueError("Unknown mode: {}".format(mode))

    with ipu_scope(_IPU_DEVICE):
      compiled_loop = ipu_compiler.compile(loop)

    if config.ipu_run_config.compile_summary:
      compile_summary_op = ipu_ops.summary_ops.ipu_compile_summary(
          "compile_summary", compiled_loop)

      # The SummarySaverHook is not added by default for evaluation,
      # so add it here if the user requested a compile summary.
      if mode == model_fn_lib.ModeKeys.EVAL and config.save_summary_steps:
        hooks.append(
            basic_session_run_hooks.SummarySaverHook(
                save_steps=config.save_summary_steps,
                output_dir=config.model_dir,
                summary_op=compile_summary_op))

    ipu_utils.move_variable_initialization_to_cpu()

    hooks.extend(wrapped_model_fn.captured_hooks)

    if mode == model_fn_lib.ModeKeys.TRAIN:
      loss, train_op = wrapped_model_fn.get_training_loss_and_op(compiled_loop)
      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op,
                                        training_hooks=hooks)
    elif mode == model_fn_lib.ModeKeys.PREDICT:
      predictions = wrapped_model_fn.get_predictions(compiled_loop)
      return model_fn_lib.EstimatorSpec(mode=mode,
                                        predictions=predictions,
                                        prediction_hooks=hooks)
    elif mode == model_fn_lib.ModeKeys.EVAL:
      loss, eval_metric_ops = wrapped_model_fn.get_evaluation_loss_and_metrics(
          compiled_loop)
      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        eval_metric_ops=eval_metric_ops,
                                        evaluation_hooks=hooks)
    else:
      raise ValueError("Unknown mode: {}".format(mode))

  return _model_fn


class _IPUEstimatorBase(estimator_lib.Estimator):
  def __init__(self,
               model_fn,
               model_dir=None,
               config=None,
               params=None,
               warm_start_from=None):
    # Base Estimator does not allow for overriding publice APIs as of June 2019
    estimator_lib.Estimator._assert_members_are_not_overridden = lambda _: None

    if config is None or not isinstance(config, ipu_run_config.RunConfig):
      raise ValueError(
          "`config` must be provided with type `ipu_run_config.RunConfig`")

    if params is not None and not isinstance(params, dict):
      raise ValueError('`params` is expected to be of type `dict`')
    if params is not None and any(k in params for k in _RESERVED_PARAMS_KEYS):
      raise ValueError('{} are reserved keys but existed in params {}.'.format(
          _RESERVED_PARAMS_KEYS, params))

    is_distributed = config._train_distribute or config._eval_distribute  # pylint: disable=protected-access
    if is_distributed and config.ipu_run_config.iterations_per_loop > 1:
      raise NotImplementedError(
          "iterations_per_loop > 1 (got {}) not supported with distribution".
          format(config.ipu_run_config.iterations_per_loop))

    super().__init__(model_fn=model_fn,
                     model_dir=model_dir,
                     config=config,
                     params=params,
                     warm_start_from=warm_start_from)

  def train(self,
            input_fn,
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None):
    """Trains a model given training data `input_fn`.

    Args:
      input_fn: A function that provides input data for training as minibatches.
        The function should return a `tf.data.Dataset` object. The outputs of
        the `Dataset` object must be a tuple `(features, labels)` where

          * `features` is a `tf.Tensor` or a dictionary of string feature name to `Tensor`
          * `labels` is a `Tensor` or a dictionary of string label name to `Tensor`

        Both `features` and `labels` are consumed by `model_fn`.
      hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
        callbacks inside the training loop.
      steps: Number of steps for which to train the model. `steps` works
        incrementally. If you call two times `train(steps=10)` then training
        occurs in total 20 steps. If you don't want to have incremental behavior
        please set `max_steps` instead. If set, `max_steps` must be `None`.
      max_steps: Number of total steps for which to train model. If set,
        `steps` must be `None`. Two calls to `train(steps=100)` means 200
        training iterations. On the other hand, two calls to `train(max_steps=100)`
        means that the second call will not do any iteration since first call did all
        100 steps.
      saving_listeners: list of `CheckpointSaverListener` objects. Used for
        callbacks that run immediately before or after checkpoint savings.

    Returns:
      `self`, for chaining.
    """
    self._validate_steps(steps)
    self._params[_INPUT_FN_KEY] = input_fn
    return super().train(input_fn=input_fn,
                         hooks=hooks,
                         steps=steps,
                         max_steps=max_steps,
                         saving_listeners=saving_listeners)

  def _convert_train_steps_to_hooks(self, steps, max_steps):
    return [
        _IPUGlobalStepCounterAndStopHook(
            self._config.ipu_run_config.iterations_per_loop, steps, max_steps)
    ]

  def _convert_eval_steps_to_hooks(self, steps):
    return self._convert_train_steps_to_hooks(steps, max_steps=None)

  def _validate_steps(self, steps):
    iterations_per_loop = self.config.ipu_run_config.iterations_per_loop
    if steps is not None and steps % iterations_per_loop != 0:
      raise ValueError(
          "steps ({}) must be a multiple of iterations_per_loop ({})".format(
              steps, iterations_per_loop))

  def _create_global_step(self, graph):
    # Overridden to make sure it is a resource variable and placed on the host,
    # while being cached on the IPU. It must be a resource variable for the
    # _validate_global_step_not_incremented() check to work, otherwise it fails
    # too early. It must be cached on the IPU in order to be readable from the
    # model function.
    graph = graph or ops.get_default_graph()
    if training_util.get_global_step(graph) is not None:
      raise ValueError('"global_step" already exists.')
    with graph.as_default() as g, g.name_scope(None), ops.device(_HOST_DEVICE):
      return variable_scope.get_variable(
          ops.GraphKeys.GLOBAL_STEP,
          shape=[],
          dtype=dtypes.int64,
          initializer=init_ops.zeros_initializer(),
          trainable=False,
          use_resource=True,
          caching_device=_IPU_DEVICE,
          aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA,
          collections=[
              ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.GLOBAL_STEP
          ])

  def _flatten_predictions(self, predictions):
    for nested_predictions in predictions:
      if isinstance(nested_predictions, dict):
        for i in range(self._extract_batch_length(nested_predictions)):
          yield {
              key: value[i]
              for key, value in six.iteritems(nested_predictions)
          }
      else:
        for prediction in nested_predictions:
          yield prediction

  def evaluate(self,
               input_fn,
               steps=None,
               hooks=None,
               checkpoint_path=None,
               name=None):
    """Evaluates the model given evaluation data `input_fn`.

    Args:
      input_fn: A function that constructs the input data for evaluation.
        The function should return a `tf.data.Dataset` object. The outputs of
        the `Dataset` object must be a tuple `(features, labels)` where

          * `features` is a `tf.Tensor` or a dictionary of string feature name to `Tensor`
          * `labels` is a `Tensor` or a dictionary of string label name to `Tensor`

        Both `features` and `labels` are consumed by `model_fn`.
      steps: Number of steps for which to evaluate model.
      hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
        callbacks inside the evaluation call.
      checkpoint_path: Path of a specific checkpoint to evaluate. If `None`, the
        latest checkpoint in `model_dir` is used.  If there are no checkpoints
        in `model_dir`, evaluation is run with newly initialized `Variables`
        instead of ones restored from checkpoint.
      name: Name of the evaluation if user needs to run multiple evaluations on
        different data sets, such as on training data vs test data. Metrics for
        different evaluations are saved in separate folders, and appear
        separately in tensorboard.

    Returns:
      A dict containing the evaluation metrics specified in `model_fn` keyed by
      name, as well as an entry `global_step` which contains the value of the
      global step for which this evaluation was performed.
    """

    self._validate_steps(steps)
    self._params[_INPUT_FN_KEY] = input_fn
    return super().evaluate(input_fn=input_fn,
                            hooks=hooks,
                            steps=steps,
                            checkpoint_path=checkpoint_path,
                            name=name)

  def predict(self,
              input_fn,
              predict_keys=None,
              hooks=None,
              checkpoint_path=None,
              yield_single_examples=True,
              num_predictions=None):
    """Yields predictions for given features.

    Args:
      input_fn: A function that constructs the features. The function should
        return a `tf.data.Dataset` object. The outputs of the `Dataset` object
        should be one of the following:

          * features: A `Tensor` or a dictionary of string feature name to
            `Tensor`. features are consumed by `model_fn`.
          * A tuple, in which case the first item is extracted as features.

      predict_keys: list of `str`, name of the keys to predict. It is used if
        the `tf.estimator.EstimatorSpec.predictions` is a `dict`. If
        `predict_keys` is used then rest of the predictions will be filtered
        from the dictionary. If `None`, returns all.
      hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
        callbacks inside the prediction call.
      checkpoint_path: Path of a specific checkpoint to predict. If `None`, the
        latest checkpoint in `model_dir` is used.  If there are no checkpoints
        in `model_dir`, prediction is run with newly initialized `Variables`
        instead of ones restored from checkpoint.
      yield_single_examples: If `False`, yields the whole batch as returned by
        the `model_fn` instead of decomposing the batch into individual
        elements. This is useful if `model_fn` returns some tensors whose first
        dimension is not equal to the batch size.
      num_predictions: If not `None`, the generator will raise `StopIteration`
        after yielding this number of predictions. This allows draining the
        generator by using :code:`list(predictions)`. If `None`, the returned
        generator is infinite and will trigger a fatal error if you try to
        consume more predictions from it than what is actually generated,
        instead of raising the `StopIteration` exception. This is caused by
        the current behaviour when requesting to run a loop on the IPU for
        more iterations than there are elements remaining in the dataset.
        In this case you cannot drain it by using :code:`list(predictions)`,
        you have to consume the expected number of elements yourself, e.g.
        using :code:`[next(predictions) for _ in range(num_predictions)]`.

    Yields:
      Evaluated values of `predictions` tensors.
    """

    self._params[_INPUT_FN_KEY] = input_fn
    predictions = super().predict(input_fn=input_fn,
                                  predict_keys=predict_keys,
                                  hooks=hooks,
                                  checkpoint_path=checkpoint_path,
                                  yield_single_examples=yield_single_examples)

    # If yield_single_examples == True, the base class has
    # already flattened the outermost iterations_per_loop
    # dimension, but we also want to flatten the batch dimension.
    # If however yield_single_examples == False, we need to
    # flatten the iterations_per_loop dimension ourselves.
    # So in both cases we need to flatten the output here.
    flat_predictions = self._flatten_predictions(predictions)

    # Raise StopIteration after num_predictions (if not None)
    return itertools.islice(flat_predictions, num_predictions)


class IPUEstimator(_IPUEstimatorBase):
  """Estimator with IPU support.

  IPUEstimator handles many of the details of running on IPUs, such as
  placement of operations and tensors, graph compilation and usage of
  data feeds. It also provides a simple way to use multiple IPUs in the
  form of either data parallelism or model parallelism.

  The data parallelism is based on graph replication. One batch from the
  dataset returned by the `input_fn` (of size `batch_size`) is sent to each
  replica, giving an effective batch size of `num_replicas * batch_size`.
  The only change needed to the `model_fn` is that the optimizer should be
  wrapped in an
  :class:`~tensorflow.python.ipu.optimizers.cross_replica_optimizer.CrossReplicaOptimizer`
  in order to average the gradients across the replicas.

  This can also be combined with distributed multi-worker training using the
  :class:`~tensorflow.python.ipu.ipu_multi_worker_strategy.IPUMultiWorkerStrategy`,
  giving a total effective batch size of
  `num_workers * num_replicas * batch_size`.

  The model parallelism supported by this class is basic sharding. Consider
  using the
  :class:`~tensorflow.python.ipu.ipu_pipeline_estimator.IPUPipelineEstimator`
  to get pipelined execution.

  For efficiency, it supports compiling a graph that contains multiple
  iterations of the training/prediction/evaluation loop, which will be
  fully executed on the IPU before yielding back to the TensorFlow
  Python runtime on the CPU.

  See https://tensorflow.org/guide/estimators for general information
  about estimators.

  Args:
    model_fn: The model function. Refer to
      https://www.tensorflow.org/guide/custom_estimators#write_a_model_function
      for details on how to write this function.
    model_dir: Directory to save model parameters, graph and etc. This can
      also be used to load checkpoints from the directory into an estimator to
      continue training a previously saved model. If `PathLike` object, the
      path will be resolved. If `None`, the model_dir in `config` will be used
      if set. If both are set, they must be same. If both are `None`, a
      temporary directory will be used.
    config: `tf.ipu.ipu_run_config.RunConfig` configuration object.
    params: `dict` of hyper parameters that will be passed into `model_fn`.
            Keys are names of parameters, values are basic python types.
    warm_start_from: Optional string filepath to a checkpoint or SavedModel to
                     warm-start from, or a `tf.estimator.WarmStartSettings`
                     object to fully configure warm-starting.  If the string
                     filepath is provided instead of a
                     `tf.estimator.WarmStartSettings`, then all variables are
                     warm-started, and it is assumed that vocabularies
                     and `tf.Tensor` names are unchanged.
  """
  def __init__(self,
               model_fn,
               model_dir=None,
               config=None,
               params=None,
               warm_start_from=None):
    # Verifies the model_fn signature according to Estimator framework.
    estimator_lib._verify_model_fn_args(model_fn, params)  # pylint: disable=protected-access

    model_function = _augment_model_fn(model_fn, _ModelFnWrapper)

    super().__init__(model_fn=model_function,
                     model_dir=model_dir,
                     config=config,
                     params=params,
                     warm_start_from=warm_start_from)
