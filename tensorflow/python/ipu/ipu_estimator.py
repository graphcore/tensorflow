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
"""IPUEstimator class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.compiler.plugin.poplar.driver.config_pb2 import IpuOptions
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import ipu_run_config
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import ops as ipu_ops
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu.ipu_outfeed_queue import IPUOutfeedMode
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.util import function_utils

_INITIAL_LOSS = 0.0
_INPUT_FN_KEY = "input_fn"

# Keys that cannot be used in the `params` dictionary passed to the
# IPUEstimator
_RESERVED_PARAMS_KEYS = [_INPUT_FN_KEY]


def _next_feed_id():
  result = "feed" + str(_next_feed_id.feed_count)
  _next_feed_id.feed_count += 1
  return result


_next_feed_id.feed_count = 0


class _IPUConfigureIPUSystemHook(session_run_hook.SessionRunHook):
  def __init__(self, config, host_device="cpu"):
    if not isinstance(config, IpuOptions):
      raise Exception("`config` must be an IpuOptions instance")
    self._config = config
    self._host_device = host_device

  def begin(self):
    ipu_utils.configure_ipu_system(self._config, self._host_device)


class _IPUInfeedInitializerSessionHook(session_run_hook.SessionRunHook):
  def __init__(self, infeed):
    self._infeed = infeed

  def after_create_session(self, session, coord):
    session.run(self._infeed.initializer)


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
    with ops.device("cpu"):
      self._increment_op = self._global_step_tensor.assign_add(
          self._iterations_per_loop)

  def after_run(self, run_context, run_values):
    global_step = run_context.session.run(self._increment_op)
    if global_step >= self._final_step:
      run_context.request_stop()


def _call_input_fn(input_fn, mode, params, config):
  input_fn_args = function_utils.fn_args(input_fn)
  kwargs = {}
  if "mode" in input_fn_args:
    kwargs["mode"] = mode
  if "params" in input_fn_args:
    kwargs["params"] = params
  if "config" in input_fn_args:
    kwargs["config"] = config
  return input_fn(**kwargs)


class _ModelFnWrapper(object):
  def __init__(self, model_fn, config, params):
    self._model_fn = model_fn
    self._config = config
    self._params = params

  def create_training_loop(self, iterations_per_loop, infeed_queue):
    def training_step(total_loss, features, labels=None):
      estimator_spec = self._call_model_fn(features, labels,
                                           model_fn_lib.ModeKeys.TRAIN)

      loss = estimator_spec.loss
      if loss is None:
        raise ValueError("EstimatorSpec must contain loss when training")

      train_op = estimator_spec.train_op
      if train_op is None:
        raise ValueError("EstimatorSpec must contain train_op when training")

      # training_step will be run by xla.compile(). xla.compile() only supports
      # tensor output while train_op can be either an operation or a tensor.
      # Even though xla.compile() automatically adds operation-typed train_op as
      # control dependency of other tensor outputs, it doesn"t do so for
      # tensor-typed train_op. Thus, we need to set it explicitly here.
      with ops.control_dependencies([train_op]):
        total_loss += math_ops.cast(loss, dtypes.float32)
        return total_loss

    def training_loop():
      return loops.repeat(iterations_per_loop,
                          training_step,
                          inputs=[_INITIAL_LOSS],
                          infeed_queue=infeed_queue)

    return training_loop

  def create_evaluation_loop(self, iterations_per_loop, infeed_queue,
                             outfeed_queue):
    def evaluation_step(total_loss, features, labels=None):
      estimator_spec = self._call_model_fn(features, labels,
                                           model_fn_lib.ModeKeys.EVAL)

      loss = estimator_spec.loss
      if loss is None:
        raise ValueError("EstimatorSpec must contain loss when evaluating")

      eval_metric_ops = estimator_spec.eval_metric_ops
      if eval_metric_ops is None:
        raise ValueError(
            "EstimatorSpec must contain eval_metric_ops when evaluating")

      update_op, value_ops = estimator_lib._extract_metric_update_ops(  # pylint: disable=protected-access
          eval_metric_ops)

      with ops.control_dependencies([update_op, loss]):
        total_loss += math_ops.cast(loss, dtypes.float32)
        outfeed = outfeed_queue.enqueue(value_ops)
        return total_loss, outfeed

    def evaluation_loop():
      return loops.repeat(iterations_per_loop,
                          evaluation_step,
                          inputs=[_INITIAL_LOSS],
                          infeed_queue=infeed_queue)

    return evaluation_loop

  def create_prediction_loop(self, iterations_per_loop, infeed_queue,
                             outfeed_queue):
    def prediction_step(features, labels=None):
      estimator_spec = self._call_model_fn(features, labels,
                                           model_fn_lib.ModeKeys.PREDICT)

      predictions = estimator_spec.predictions
      if predictions is None:
        raise ValueError(
            "EstimatorSpec must contain predictions when predicting")

      outfeed = outfeed_queue.enqueue(predictions)
      return outfeed

    def prediction_loop():
      return loops.repeat(iterations_per_loop,
                          prediction_step,
                          infeed_queue=infeed_queue)

    return prediction_loop

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

    if not isinstance(estimator_spec, model_fn_lib.EstimatorSpec):
      raise ValueError("`model_fn` must return `tf.estimator.EstimatorSpec`")

    return estimator_spec


def _augment_model_fn(model_fn, iterations_per_loop):
  """Returns a new model_fn, which wraps the IPU support."""

  def _model_fn(features, labels, mode, config, params):  # pylint: disable=unused-argument
    input_fn = params[_INPUT_FN_KEY]
    dataset = _call_input_fn(input_fn, mode, params, config)
    if not isinstance(dataset, dataset_ops.Dataset):
      raise ValueError("input_fn must return Dataset")

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, _next_feed_id())

    hooks = [
        _IPUInfeedInitializerSessionHook(infeed_queue),
    ]

    if config.ipu_run_config.ipu_options is None:
      if config.ipu_run_config.compile_summary:
        logging.warning(
            "Compile summary enabled but IpuOptions is None. No profile will be generated"
        )

    if config.ipu_run_config.ipu_options is not None:
      ipu_options = config.ipu_run_config.ipu_options
      hooks += [_IPUConfigureIPUSystemHook(ipu_options, host_device="cpu")]

    model_fn_wrapper = _ModelFnWrapper(model_fn, config, params)

    if mode == model_fn_lib.ModeKeys.TRAIN:
      loop = model_fn_wrapper.create_training_loop(iterations_per_loop,
                                                   infeed_queue)
    elif mode == model_fn_lib.ModeKeys.EVAL:
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(
          "eval_" + _next_feed_id(), outfeed_mode=IPUOutfeedMode.LAST)
      loop = model_fn_wrapper.create_evaluation_loop(iterations_per_loop,
                                                     infeed_queue,
                                                     outfeed_queue)
    elif mode == model_fn_lib.ModeKeys.PREDICT:
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(
          "predict_" + _next_feed_id(), outfeed_mode=IPUOutfeedMode.ALL)
      loop = model_fn_wrapper.create_prediction_loop(iterations_per_loop,
                                                     infeed_queue,
                                                     outfeed_queue)
    else:
      raise ValueError("Unknown mode: {}".format(mode))

    with ipu_scope("/device:IPU:0"):
      compiled_loop = ipu_compiler.compile(loop)

    if config.ipu_run_config.compile_summary:
      ipu_ops.summary_ops.ipu_compile_summary("compile_summary", compiled_loop)

    ipu_utils.move_variable_initialization_to_cpu()

    if mode in (model_fn_lib.ModeKeys.TRAIN, model_fn_lib.ModeKeys.EVAL):
      train_op = total_loss = compiled_loop[0]
      loss = total_loss / iterations_per_loop
      predictions = None
    else:
      assert mode == model_fn_lib.ModeKeys.PREDICT
      train_op = None
      loss = None
      with ops.control_dependencies([compiled_loop]):
        predictions = outfeed_queue.dequeue()

    eval_metric_ops = {}
    if mode == model_fn_lib.ModeKeys.EVAL:
      dequeue_op = outfeed_queue.dequeue()
      for metric_name, outfed_tensor in six.iteritems(dequeue_op):
        # No op as update-op since updating is done inside the loop
        eval_metric_ops[metric_name] = (outfed_tensor,
                                        control_flow_ops.no_op())

    return model_fn_lib.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op,
                                      training_hooks=hooks,
                                      evaluation_hooks=hooks,
                                      prediction_hooks=hooks,
                                      eval_metric_ops=eval_metric_ops,
                                      predictions=predictions)

  return _model_fn


class IPUEstimator(estimator_lib.Estimator):
  """Estimator with IPU support."""

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

    # Verifies the model_fn signature according to Estimator framework.
    estimator_lib._verify_model_fn_args(model_fn, params)  # pylint: disable=protected-access

    if config.ipu_run_config.ipu_options is not None:
      dev = config.ipu_run_config.ipu_options.device_config
      if len(dev) > 1 or (len(dev) == 1 and dev[0].auto_count > 1):
        raise NotImplementedError("Only one IPU is currently supported")

    if params is not None and not isinstance(params, dict):
      raise ValueError('`params` is expected to be of type `dict`')
    if params is not None and any(k in params for k in _RESERVED_PARAMS_KEYS):
      raise ValueError('{} are reserved keys but existed in params {}.'.format(
          _RESERVED_PARAMS_KEYS, params))

    model_function = _augment_model_fn(
        model_fn, config.ipu_run_config.iterations_per_loop)

    super(IPUEstimator, self).__init__(model_fn=model_function,
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
    self._params[_INPUT_FN_KEY] = input_fn
    return super(IPUEstimator, self).train(input_fn=input_fn,
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

  def evaluate(self,
               input_fn,
               steps=None,
               hooks=None,
               checkpoint_path=None,
               name=None):
    self._params[_INPUT_FN_KEY] = input_fn
    return super(IPUEstimator, self).evaluate(input_fn=input_fn,
                                              hooks=hooks,
                                              steps=steps,
                                              checkpoint_path=checkpoint_path,
                                              name=name)

  def predict(self,
              input_fn,
              predict_keys=None,
              hooks=None,
              checkpoint_path=None,
              yield_single_examples=True):
    """The returned generator will block forever if you try to consume
    more elements than what is generated, instead of raising the regular
    `StopIteration` exception. This is caused by the current behaviour
    when requesting to run a loop on the IPU for more iterations than there
    are elements remaining in the dataset. So you cannot simply drain it by
    using `list(predictions)`, you have to consume the expected number of
    elements, e.g. using `[next(predictions) for _ in range(num_examples)]`."""

    self._params[_INPUT_FN_KEY] = input_fn
    predictions = super(IPUEstimator, self).predict(
        input_fn=input_fn,
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
