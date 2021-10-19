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
from tensorflow.python.ipu import config as ipu_config
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import ipu_run_config
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import ops as ipu_ops
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu.ipu_multi_worker_strategy import IPUMultiWorkerStrategyV1
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
_BATCH_SIZE_KEY = "batch_size"
_ASSIGN_ADD_OP = "AssignAddVariableOp"
_CROSS_REPLICA_SUM_OP = "IpuCrossReplicaSum"
_CROSS_REPLICA_MEAN_OP = "IpuCrossReplicaMean"
_RESOURCE_UPDATE_OP = "ResourceUpdate"
_HOST_DEVICE = "/device:CPU:0"

# Keys that cannot be used in the `params` dictionary passed to the
# IPUEstimator
_RESERVED_PARAMS_KEYS = [_INPUT_FN_KEY]


def _validate_function_call_spec(call_spec, name, allow_dict_arg=False):
  if call_spec is not None:
    if not isinstance(call_spec, tuple):
      raise TypeError("`{}` must be a tuple".format(name))
    if len(call_spec) != 2:
      raise ValueError("`{}` must have two elements".format(name))
    if not callable(call_spec[0]):
      raise TypeError("first element in `{}` must be callable".format(name))

    allowed_arg_types = (list,)
    if allow_dict_arg:
      allowed_arg_types += (dict,)
    if not isinstance(call_spec[1], allowed_arg_types):
      raise TypeError("second element in `{}` must be a {}".format(
          name, " or ".join(t.__name__ for t in allowed_arg_types)))


class IPUEstimatorSpec(
    collections.namedtuple('IPUEstimatorSpec', [
        'mode', 'predictions', 'loss', 'train_op', 'eval_metric_ops',
        'eval_metrics', 'host_call', 'training_hooks', 'evaluation_hooks',
        'prediction_hooks'
    ])):
  """Ops and objects returned from a `model_fn` and passed to `IPUEstimator`.

  This is very similar to `EstimatorSpec`, with the addition of two extra
  arguments: `eval_metrics` and `host_call`. If neither of those arguments
  are needed, an `EstimatorSpec` can be passed to the `IPUEstimator` instead.

  `eval_metrics` is a tuple of a (`function`, `tensors`), where `tensors` is
  either a list of `tf.Tensor` or a dict from strings to `tf.Tensor`, that is
  passed to the function. The function runs on the CPU and returns a dict of
  metrics. The tensors are transferred from the IPU to the CPU host and passed
  to the function.

  Exactly one of `eval_metrics` and `eval_metric_ops` must be provided during
  evaluation. The major difference between the two is that while the
  `eval_metric_ops` will execute directly on the IPU, the `eval_metrics` will
  execute on the CPU host using the provided function. Example:

  .. code-block:: python

    def my_metrics_fn(features, labels):
      return {
          "accuracy": tf.metrics.accuracy(labels, features),
          "precision": tf.metrics.precision(labels, features),
          "recall": tf.metrics.recall(labels, features),
      }

    eval_metrics = (my_metrics_fn, [features, labels])
    spec = IPUEstimatorSpec(mode, loss=loss, eval_metrics=eval_metrics)

  `host_call` is a tuple of a function and a list of tensors to pass to that
  function. `host_call` only works for training and is executed on the CPU for
  every training step. The tensors are transferred from the IPU to the CPU host
  and passed to the function.

  This functionality can be used for e.g. doing all-reduce of the gradients and
  weight updates on the host during distributed training with the
  `IPUMultiWorkerStrategyV1`. Example:

  .. code-block:: python

    def my_host_fn(*host_gradients):
      # This will all-reduce the gradients and update the weights on the host.
      return optimizer.apply_gradients(zip(host_gradients, variables))

    train_op = tf.identity(loss)
    grads_and_vars = optimizer.compute_gradients(loss, var_list=variables)
    gradients = [g for (g, _) in grads_and_vars]
    host_call = (my_host_fn, gradients)

    spec = IPUEstimatorSpec(mode=mode,
                            loss=loss,
                            train_op=train_op,
                            host_call=host_call)

  See full example: :any:`distributed_training`.

  The various hooks (`training_hooks, `evaluation_hooks`, `prediction_hooks`)
  support instances of `tf.estimator.SessionRunHook`. To log tensor values from
  within the `model_fn`, use the
  :class:`~tensorflow.python.ipu.ipu_session_run_hooks.IPULoggingTensorHook`.

  For documentation of the remaining arguments, see `EstimatorSpec`.
  """
  def __new__(cls,
              mode,
              predictions=None,
              loss=None,
              train_op=None,
              eval_metric_ops=None,
              eval_metrics=None,
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

    _validate_function_call_spec(host_call, "host_call")

    _validate_function_call_spec(eval_metrics,
                                 "eval_metrics",
                                 allow_dict_arg=True)

    return super().__new__(cls,
                           mode=mode,
                           predictions=predictions,
                           loss=loss,
                           train_op=train_op,
                           eval_metric_ops=eval_metric_ops,
                           eval_metrics=eval_metrics,
                           host_call=host_call,
                           training_hooks=training_hooks,
                           evaluation_hooks=evaluation_hooks,
                           prediction_hooks=prediction_hooks)


class _IPUConfigureIPUSystemHook(session_run_hook.SessionRunHook):
  def __init__(self, config, host_device=_HOST_DEVICE):
    if not isinstance(config.ipu_options, (IpuOptions, ipu_config.IPUConfig)):
      raise Exception("`config.ipu_options` must be an IPUConfig or IpuOptions"
                      " instance")
    self._config = config.ipu_options
    if isinstance(self._config, ipu_config.IPUConfig):
      self._config = self._config._create_protobuf()  # pylint: disable=protected-access
    self._run_config = config
    self._host_device = host_device

  def begin(self):
    ipu_utils.configure_ipu_system(self._config, self._host_device)

    if self._config.device_config[self._run_config.ordinal].cfg_index:
      num_configured_devices = ipu_utils.get_num_of_ipus_in_device(
          '/device:IPU:{}'.format(self._run_config.ordinal))

      num_devices = self._run_config.num_shards * self._run_config.num_replicas

      if num_devices != num_configured_devices:
        raise ValueError('`IPURunConfig` configured with {} devices'
                         ' ({} num_replicas times {} num_shards),'
                         ' but `IpuOptions` configured with {} devices'.format(
                             num_devices, self._run_config.num_replicas,
                             self._run_config.num_shards,
                             num_configured_devices))


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
  def has_cross_replica_reduce_op(g):
    return any(
        op.type == _CROSS_REPLICA_SUM_OP or op.type == _CROSS_REPLICA_MEAN_OP
        for op in g.get_operations())

  graph = ops.get_default_graph()
  if has_cross_replica_reduce_op(graph):
    return

  # Also check inside the resource update `FuncGraph` if there is one.
  for op in graph.get_operations():
    if op.type == _RESOURCE_UPDATE_OP:
      resource_update_graph = graph._get_function(  # pylint: disable=protected-access
          op.get_attr("to_apply").name).graph
      if has_cross_replica_reduce_op(resource_update_graph):
        return

  raise ValueError(
      "This is not a valid replicated training graph because no " +
      _CROSS_REPLICA_SUM_OP + " or " + _CROSS_REPLICA_MEAN_OP +
      "operations were found. Did you remember to use the " +
      "`tensorflow.python.ipu.optimizers.CrossReplicaOptimizer`?")


def _add_send_to_host_ops(tensors, ipu_device):
  """Returns attributes for matching recv ops"""
  recv_ops_attrs = []

  for tensor in tensors:
    model_fn_lib._check_is_tensor_or_operation(  # pylint: disable=protected-access
        tensor, "`host_call` argument")

    attrs = dict(tensor_name=tensor.name,
                 send_device=ipu_device,
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
  def need_outfeed(mode):
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
    self._ipu_device = "/device:IPU:{}".format(config.ipu_run_config.ordinal)
    self._captured_hooks = []
    self._captured_host_call_fn = None
    self._captured_host_call_args = None
    self._captured_eval_metrics_fn = None

  @staticmethod
  def need_outfeed(mode):
    # No outfeed for training
    return mode != model_fn_lib.ModeKeys.TRAIN

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
      assert self._captured_host_call_fn is None, \
          "Can only capture host_call once"
      self._captured_host_call_fn, tensors = host_call
      self._captured_host_call_args = _add_send_to_host_ops(
          tensors, self._ipu_device)

  def _capture_eval_metrics_fn(self, metrics_fn):
    assert metrics_fn is not None
    assert self._captured_eval_metrics_fn is None, \
        "Can only capture eval_metrics_fn once"
    self._captured_eval_metrics_fn = metrics_fn

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
      eval_metrics = getattr(estimator_spec, "eval_metrics", None)
      if not eval_metric_ops and not eval_metrics:
        raise ValueError(
            "EstimatorSpec must contain either eval_metric_ops or "
            "eval_metrics when evaluating")
      if eval_metric_ops and eval_metrics:
        raise ValueError(
            "EstimatorSpec cannot contain both eval_metric_ops and "
            "eval_metrics")

      self._capture_hooks(estimator_spec.evaluation_hooks)

      if eval_metric_ops:
        outfeed_values = _extract_metric_values(eval_metric_ops)
      else:
        metrics_fn, outfeed_values = eval_metrics
        self._capture_eval_metrics_fn(metrics_fn)

      total_loss += math_ops.cast(loss, dtypes.float32)
      outfeed = self._outfeed_queue.enqueue(outfeed_values)
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
      if self._captured_eval_metrics_fn is not None:
        # Calculate metrics on the host. Control dependency on the loop needed
        # since the metric *ops* on the host must see all the enqueued inputs.
        # The metric *tensors* on the host are idempotent and will not trigger
        # another execution of the dequeue op when evaluated later.
        with ops.control_dependencies(compiled_evaluation_loop):
          inputs = self._outfeed_queue.dequeue(wait_for_completion=True)

        args, kwargs = loops._body_arguments(inputs)  # pylint: disable=protected-access
        metric_ops = self._captured_eval_metrics_fn(*args, **kwargs)
      else:
        # Metrics already calculated on IPU. Aggregate on the host. We can
        # *not* have a control dependency on the loop here as the metric
        # tensors must be idempotent, i.e. they must support evaluation
        # without triggering a new execution of the dequeue op, and our
        # pass-through metric tensors below have a data dependency on the
        # dequeue op. The metric tensors are evaluated in a separate
        # execution so they are guaranteed to see all the enqueued inputs.
        metrics = self._outfeed_queue.dequeue(wait_for_completion=True)

        metric_ops = {}
        for metric_name, metric_tensor in six.iteritems(metrics):
          # The outfeed outputs all values, but we only need the last one (the
          # most recent aggregated value) when they are calculated on IPU.
          last_metric_tensor = metric_tensor[-1]
          # For replicated graphs the tensor will have an additional replica
          # dimension, so we reduce over this dimension (if it exists).
          # Note: mean is not always correct, e.g. for root_mean_squared_error,
          # workaround is to use `eval_metrics` on host to get correct aggregation.
          # Use no-op as the update_op since updating is done inside the loop.
          metric_ops[metric_name] = (math_ops.reduce_mean(last_metric_tensor),
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
        predictions = self._outfeed_queue.dequeue(wait_for_completion=True)
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
  if isinstance(strategy, IPUMultiWorkerStrategyV1):
    return strategy.extended._make_input_context()  # pylint: disable=protected-access
  return None


def _augment_model_fn(model_fn, wrapper_class, ipu_device):
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

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, prefetch_depth=config.ipu_run_config.prefetch_depth)
    hooks.append(_IPUInfeedLifecycleHook(infeed_queue))

    if not wrapper_class.need_outfeed(mode):
      outfeed_queue = None
    else:
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(
          outfeed_mode=ipu_outfeed_queue.IPUOutfeedMode.ALL)
      hooks.append(_IPUOutfeedLifecycleHook(outfeed_queue))

    if config.ipu_run_config.ipu_options is not None:
      hooks.append(
          _IPUConfigureIPUSystemHook(config.ipu_run_config,
                                     host_device=_HOST_DEVICE))

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

    with ipu_scope(ipu_device):
      compiled_loop = ipu_compiler.compile(loop)

    if config.ipu_run_config.compile_summary:
      raise NotImplementedError(
          "Generating compilation summaries for the IPUEstimator through"
          " IPURunConfig.compile_summary is deprecated, is non-functional and"
          " will be removed in a future release. Use the PopVision suite of"
          " analysis tools to profile IPU programs.")

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


def _calc_batch_size(global_batch_size, num_workers, num_replicas, name):
  if global_batch_size is None:
    return None

  if global_batch_size < 1:
    raise ValueError("{} (got {}) must be positive".format(
        name, global_batch_size))

  batch_size, remainder = divmod(global_batch_size, num_workers * num_replicas)

  if remainder != 0:
    raise ValueError(
        "{} (got {}) must be divisible by num_workers * num_replicas ({} * {})"
        .format(name, global_batch_size, num_workers, num_replicas))

  return batch_size


class _IPUEstimatorBase(estimator_lib.Estimator):
  def __init__(self,
               model_fn,
               model_dir=None,
               config=None,
               params=None,
               warm_start_from=None,
               train_batch_size=None,
               eval_batch_size=None,
               predict_batch_size=None):
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

    self._any_batch_size_provided = ((train_batch_size is not None)
                                     or (eval_batch_size is not None)
                                     or (predict_batch_size is not None))

    if (self._any_batch_size_provided and params is not None
        and _BATCH_SIZE_KEY in params):
      raise ValueError(
          "{} cannot be passed in params when a batch size argument is passed".
          format(_BATCH_SIZE_KEY))

    # pylint: disable=protected-access
    num_train_workers = config._train_distribute.num_replicas_in_sync if \
        config._train_distribute else 1
    num_eval_workers = config._eval_distribute.num_replicas_in_sync if \
        config._eval_distribute else 1
    # pylint: enable=protected-access

    num_replicas = config.ipu_run_config.num_replicas

    self._ipu_device = "/device:IPU:{}".format(config.ipu_run_config.ordinal)

    self._batch_size_for_train = _calc_batch_size(train_batch_size,
                                                  num_train_workers,
                                                  num_replicas,
                                                  "train_batch_size")

    self._batch_size_for_eval = _calc_batch_size(eval_batch_size,
                                                 num_eval_workers,
                                                 num_replicas,
                                                 "eval_batch_size")

    self._batch_size_for_predict = _calc_batch_size(predict_batch_size, 1,
                                                    num_replicas,
                                                    "predict_batch_size")

    self._validate_config(config)

    super().__init__(model_fn=model_fn,
                     model_dir=model_dir,
                     config=config,
                     params=params,
                     warm_start_from=warm_start_from)

  def _setup_params(self, input_fn, batch_size):
    self._params[_INPUT_FN_KEY] = input_fn

    if self._any_batch_size_provided:
      if batch_size is not None:
        if "params" not in function_utils.fn_args(input_fn):
          raise ValueError(
              "input_fn must have params argument to receive params['{}']".
              format(_BATCH_SIZE_KEY))

        self._params[_BATCH_SIZE_KEY] = batch_size
      else:
        # Remove any value left from previous call.
        self._params.pop(_BATCH_SIZE_KEY, None)

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
    self._setup_params(input_fn, self._batch_size_for_train)

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

  def _validate_config(self, config):
    is_distributed = config._train_distribute or config._eval_distribute  # pylint: disable=protected-access
    if is_distributed and config.ipu_run_config.iterations_per_loop > 1:
      raise NotImplementedError(
          "iterations_per_loop > 1 (got {}) not supported with distribution".
          format(config.ipu_run_config.iterations_per_loop))

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
          caching_device=self._ipu_device,
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
    self._setup_params(input_fn, self._batch_size_for_eval)

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
    self._setup_params(input_fn, self._batch_size_for_predict)

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
  wrapped in a
  :class:`~tensorflow.python.ipu.optimizers.CrossReplicaOptimizer`
  in order to average the gradients across the replicas.

  This can also be combined with distributed multi-worker training using the
  :class:`~tensorflow.python.ipu.ipu_multi_worker_strategy.IPUMultiWorkerStrategyV1`,
  giving a total effective batch size of
  `num_workers * num_replicas * batch_size`.

  The desired global batch size can be passed as `train_batch_size`,
  `eval_batch_size` and `predict_batch_size`, and the local batch size will be
  calculated based on the number of replicas and the number of distributed
  workers and passed to the `input_fn` and `model_fn` in
  `params['batch_size']`. If the `input_fn` returns a dataset batched with
  `dataset.batch(params['batch_size'], drop_remainder=True)`, the global batch
  size will be as desired.

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
      https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/custom_estimators.md#write-a-model-function
      for details on how to write this function.
    model_dir: Directory to save model parameters, graph and etc. This can
      also be used to load checkpoints from the directory into an estimator to
      continue training a previously saved model. If `PathLike` object, the
      path will be resolved. If `None`, the model_dir in `config` will be used
      if set. If both are set, they must be same. If both are `None`, a
      temporary directory will be used.
    config: A :class:`~tensorflow.python.ipu.ipu_run_config.RunConfig` object.
    params: `dict` of hyper parameters that will be passed into `model_fn`.
            Keys are names of parameters, values are basic python types.
    warm_start_from: Optional string filepath to a checkpoint or SavedModel to
                     warm-start from, or a `tf.estimator.WarmStartSettings`
                     object to fully configure warm-starting.  If the string
                     filepath is provided instead of a
                     `tf.estimator.WarmStartSettings`, then all variables are
                     warm-started, and it is assumed that vocabularies
                     and `tf.Tensor` names are unchanged.
    train_batch_size: If not None, an int representing the global training
      batch size. This global batch size is transformed to a local batch size
      passed as `params['batch_size']` to the `input_fn` and `model_fn` during
      training. Must be divisible by the number of replicas multiplied by the
      number of distributed workers.
    eval_batch_size: If not None, an int representing the global evaluation
      batch size. Same behaviour as train_batch_size, only during evaluation.
    predict_batch_size: If not None, an int representing the global prediction
      batch size. Same behaviour as train_batch_size, only during prediction.
  """
  def __init__(self,
               model_fn,
               model_dir=None,
               config=None,
               params=None,
               warm_start_from=None,
               train_batch_size=None,
               eval_batch_size=None,
               predict_batch_size=None):
    # Verifies the model_fn signature according to Estimator framework.
    estimator_lib._verify_model_fn_args(model_fn, params)  # pylint: disable=protected-access

    ipu_device = "/device:IPU:{}".format(config.ipu_run_config.ordinal)

    model_function = _augment_model_fn(model_fn, _ModelFnWrapper, ipu_device)

    super().__init__(model_fn=model_function,
                     model_dir=model_dir,
                     config=config,
                     params=params,
                     warm_start_from=warm_start_from,
                     train_batch_size=train_batch_size,
                     eval_batch_size=eval_batch_size,
                     predict_batch_size=predict_batch_size)
