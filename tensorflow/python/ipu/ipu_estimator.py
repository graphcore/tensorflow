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

from tensorflow.compiler.plugin.poplar.driver.config_pb2 import IpuOptions
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import ipu_run_config
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import ops as ipu_ops
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.util import function_utils

_INITIAL_LOSS = 1e7


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


class _IPUInfeedOutfeedSessionHook(session_run_hook.SessionRunHook):
  def __init__(self, infeed, outfeed=None):
    self._infeed = infeed
    self._outfeed = outfeed

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


def _call_model_fn(model_fn, features, labels, mode, params, config):
  model_fn_args = function_utils.fn_args(model_fn)
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
    kwargs["params"] = params
  if "config" in model_fn_args:
    kwargs["config"] = config
  return model_fn(features=features, **kwargs)


def _augment_model_fn(model_fn, input_fn, iterations_per_loop):
  """Returns a new model_fn, which wraps the IPU support."""

  def _model_fn(features, labels, mode, config, params):
    assert mode == model_fn_lib.ModeKeys.TRAIN

    dataset = _call_input_fn(input_fn, mode, params, config)
    if not isinstance(dataset, dataset_ops.Dataset):
      raise ValueError("input_fn must return Dataset")

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, _next_feed_id())
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(
        _next_feed_id(), outfeed_mode=config.ipu_run_config.outfeed_mode)

    training_hooks = [
        _IPUInfeedOutfeedSessionHook(infeed_queue, outfeed_queue),
    ]

    if config.ipu_run_config.ipu_options is None:
      if config.ipu_run_config.compile_summary:
        logging.warning(
            "Compile summary enabled but IpuOptions is None. No profile will be generated"
        )

    if config.ipu_run_config.ipu_options is not None:
      ipu_options = config.ipu_run_config.ipu_options
      training_hooks += [
          _IPUConfigureIPUSystemHook(ipu_options, host_device="cpu")
      ]

    def training_step(loss, features, labels=None):
      del loss  # unused; required in function signature.

      estimator_spec = _call_model_fn(model_fn, features, labels, mode, params,
                                      config)
      if not isinstance(estimator_spec, model_fn_lib.EstimatorSpec):
        raise ValueError("`model_fn` must return `tf.estimator.EstimatorSpec`")

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
      outfeed = outfeed_queue.enqueue(loss)
      with ops.control_dependencies([train_op]):
        return (array_ops.identity(loss, name="model_fn_loss"), outfeed)

    def training_loop():
      return loops.repeat(iterations_per_loop,
                          training_step,
                          inputs=[_INITIAL_LOSS],
                          infeed_queue=infeed_queue)

    with ipu_scope("/device:IPU:0"):
      compiled_training_loop = ipu_compiler.compile(training_loop)

    train_op = compiled_training_loop[0]
    with ops.control_dependencies([train_op]):
      loss = outfeed_queue.dequeue()

    ipu_utils.move_variable_initialization_to_cpu()

    if config.ipu_run_config.compile_summary:
      ipu_ops.summary_ops.ipu_compile_summary("compile_summary", train_op)

    return model_fn_lib.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op,
                                      training_hooks=training_hooks)

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

    self._model_fn_augmented = False

    super(IPUEstimator, self).__init__(model_fn=model_fn,
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
    if not self._model_fn_augmented:
      self._model_fn = _augment_model_fn(
          self._model_fn, input_fn,
          self._config.ipu_run_config.iterations_per_loop)
      self._model_fn_augmented = True

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

  def evaluate(self,
               input_fn,
               steps=None,
               hooks=None,
               checkpoint_path=None,
               name=None):
    raise NotImplementedError()

  def predict(self,
              input_fn,
              predict_keys=None,
              hooks=None,
              checkpoint_path=None,
              yield_single_examples=True):
    raise NotImplementedError()
