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
IPUPipelineEstimator
~~~~~~~~~~~~~~~~~~~~
"""

import collections

from tensorflow.python import ops
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.ipu import ipu_estimator
from tensorflow.python.ipu import loops
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer as ga
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import function_utils

_HOST_DEVICE = ipu_estimator._HOST_DEVICE  # pylint: disable=protected-access


class IPUPipelineEstimatorSpec(
    collections.namedtuple('IPUPipelineEstimatorSpec', [
        'mode',
        'computational_stages',
        'gradient_accumulation_count',
        'eval_metrics_fn',
        'optimizer_function',
        'device_mapping',
        'loss_accumulator_dtype',
        'training_hooks',
        'evaluation_hooks',
        'prediction_hooks',
        'reduction_method',
        'pipeline_op_kwargs',
    ])):
  """Ops and objects returned from a `model_fn` and passed to
  :class:`.IPUPipelineEstimator`."""
  def __new__(cls,
              mode,
              computational_stages,
              gradient_accumulation_count=None,
              eval_metrics_fn=None,
              optimizer_function=None,
              device_mapping=None,
              loss_accumulator_dtype=None,
              training_hooks=None,
              evaluation_hooks=None,
              prediction_hooks=None,
              reduction_method=ga.GradientAccumulationReductionMethod.MEAN,
              **pipeline_op_kwargs):
    """Creates a validated `IPUPipelineEstimatorSpec` instance.

    Depending on the value of `mode`, different arguments are required. Namely

    * For `mode == ModeKeys.TRAIN`: the `optimizer_function` is required.
    * For `mode == ModeKeys.EVAL`: the `eval_metrics_fn` is required.

    Refer to the :mod:`~tensorflow.python.ipu.pipelining_ops`
    documentation for more details about pipelining.

    Note that the pipeline keyword argument `accumulate_outfeed` cannot be
    passed to the `IPUPipelineEstimatorSpec`, since the `IPUPipelineEstimator`
    uses it internally to accumulate the loss when training.

    Args:
      mode: A `ModeKeys`. Specifies if this is training, evaluation or
        prediction.
      computational_stages: a list of Python functions, where each function
        represents a computational pipeline stage. The function takes the
        outputs of the previous pipeline state as its inputs.
      gradient_accumulation_count: the number of times each pipeline stage will
        be executed.
      eval_metrics_fn: a Python function which takes the output of the
        last computational stage as parameters and returns a dict of evaluation
        metrics. The dict must contain a a loss tensor value with the key
        "loss". This function will be called on the host.
      optimizer_function: a Python function which takes the output of the
        last computational stage as parameters and returns an instance of
        :class:`~tensorflow.python.ipu.pipelining_ops.OptimizerFunctionOutput`
        in order to generate the back-propagation and weight-update parts of the
        model suitable for training.
      device_mapping: optional stage to IPU mapping override.
      loss_accumulator_dtype: When training, the loss is accumulated during
        pipeline execution onto a buffer. Use this to set the data type of the
        buffer to, for example, avoid overflow. By default (`None`), the buffer
        is the same data type as the loss.
      training_hooks: List of instances of `tf.estimator.SessionRunHook` used
        during training.
      evaluation_hooks: List of instances of `tf.estimator.SessionRunHook` used
        during evaluation.
      prediction_hooks: List of instances of `tf.estimator.SessionRunHook` used
        during prediction.
      reduction_method: Reduction method to use when accumulating gradients.
        During the iterations in each optimizer step, the computed gradients
        can either be directly summed up or scaled such that we compute a mean
        of all gradients for each variable. Computing a mean avoids potential
        issues with overflow during accumulation especially when using
        float16, but gives smaller gradients and might require adjusting
        the learning-rate accordingly.
        Defaults to `GradientAccumulationReductionMethod.SUM`
        (see :class:`~tensorflow.python.ipu.optimizers.GradientAccumulationReductionMethod`)  # pylint: disable=line-too-long
      pipeline_op_kwargs: All remaining keyword arguments are forwarded to
        :func:`~tensorflow.python.ipu.pipelining_ops.pipeline`.

    Returns:
      A validated `IPUPipelineEstimatorSpec` object.

    Raises:
      ValueError: If validation fails.
    """

    if mode == model_fn_lib.ModeKeys.TRAIN and not optimizer_function:
      raise ValueError("`IPUPipelineEstimatorSpec` must contain "
                       "`optimizer_function` when training")

    if mode == model_fn_lib.ModeKeys.EVAL and not eval_metrics_fn:
      raise ValueError("`IPUPipelineEstimatorSpec` must contain "
                       "`eval_metrics_fn` when evaluating")

    if not gradient_accumulation_count:
      raise ValueError("`IPUPipelineEstimatorSpec` must contain "
                       "`gradient_accumulation_count`")

    # Do not allow setting of the `accumulate_outfeed` pipeline kwarg, since the
    # estimator uses it internally.
    if "accumulate_outfeed" in pipeline_op_kwargs:
      raise ValueError("The `accumulate_outfeed` pipeline keyword argument"
                       " cannot be passed to the `IPUPipelineEstimatorSpec`.")

    return super().__new__(
        cls,
        mode=mode,
        computational_stages=computational_stages,
        eval_metrics_fn=eval_metrics_fn,
        gradient_accumulation_count=gradient_accumulation_count,
        optimizer_function=optimizer_function,
        device_mapping=device_mapping,
        loss_accumulator_dtype=loss_accumulator_dtype,
        training_hooks=training_hooks,
        evaluation_hooks=evaluation_hooks,
        prediction_hooks=prediction_hooks,
        reduction_method=reduction_method,
        pipeline_op_kwargs=pipeline_op_kwargs)


class _ModelFnPipelineWrapper(ipu_estimator._ModelFnWrapperBase):  # pylint: disable=protected-access
  def __init__(self, model_fn, config, params, infeed_queue, outfeed_queue):
    self._model_fn = model_fn
    self._config = config
    self._params = params
    self._infeed_queue = infeed_queue
    self._outfeed_queue = outfeed_queue
    self._captured_eval_metrics_fn = None
    self._captured_gradient_accumulation_count = None
    self._captured_hooks = []

  @staticmethod
  def need_outfeed(mode):  # pylint: disable=unused-argument
    return True

  def _capture_hooks(self, hooks):
    if hooks:
      assert not self._captured_hooks, "Can only capture hooks once"
      self._captured_hooks = hooks

  @property
  def captured_hooks(self):
    return self._captured_hooks

  def _calc_repeat_count(self, spec):
    iterations_per_loop = self._config.ipu_run_config.iterations_per_loop

    if iterations_per_loop < spec.gradient_accumulation_count:
      raise ValueError(
          ("`IPURunConfig.iterations_per_loop` (got {}) cannot be less than "
           "`gradient_accumulation_count` (got {})").format(
               iterations_per_loop, spec.gradient_accumulation_count))

    repeat_count, remainder = divmod(iterations_per_loop,
                                     spec.gradient_accumulation_count)

    if remainder != 0:
      raise ValueError(
          ("`IPURunConfig.iterations_per_loop` (got {}) must be divisible by "
           "`gradient_accumulation_count` (got {})").format(
               iterations_per_loop, spec.gradient_accumulation_count))

    return repeat_count

  def create_training_loop(self):
    def training_pipeline():
      spec = self._call_model_fn(model_fn_lib.ModeKeys.TRAIN)

      assert not self._captured_gradient_accumulation_count
      self._captured_gradient_accumulation_count = \
          spec.gradient_accumulation_count

      self._capture_hooks(spec.training_hooks)

      return pipelining_ops.pipeline(
          infeed_queue=self._infeed_queue,
          outfeed_queue=self._outfeed_queue,
          computational_stages=spec.computational_stages,
          gradient_accumulation_count=spec.gradient_accumulation_count,
          repeat_count=self._calc_repeat_count(spec),
          optimizer_function=spec.optimizer_function,
          device_mapping=spec.device_mapping,
          outfeed_loss=True,
          accumulate_outfeed=True,
          accumulate_outfeed_dtype=spec.loss_accumulator_dtype,
          reduction_method=spec.reduction_method,
          name="ipu_pipeline_estimator_train",
          **spec.pipeline_op_kwargs)

    return training_pipeline

  def get_training_loss_and_op(self, compiled_training_loop):
    with ops.device(_HOST_DEVICE):
      with ops.control_dependencies([compiled_training_loop]):
        loss = self._outfeed_queue.dequeue(wait_for_completion=True)

      assert self._captured_gradient_accumulation_count
      # Reduce loss over all dimensions, then normalise by ga-count since
      # the loss has been accumulated inside the pipeline.
      loss = math_ops.reduce_mean(math_ops.cast(
          loss, dtypes.float32)) / self._captured_gradient_accumulation_count

    train_op = compiled_training_loop

    return loss, train_op

  def create_evaluation_loop(self):
    def evaluation_pipeline():
      spec = self._call_model_fn(model_fn_lib.ModeKeys.EVAL)

      assert not self._captured_eval_metrics_fn
      assert spec.eval_metrics_fn
      self._captured_eval_metrics_fn = spec.eval_metrics_fn

      self._capture_hooks(spec.evaluation_hooks)

      return pipelining_ops.pipeline(
          infeed_queue=self._infeed_queue,
          outfeed_queue=self._outfeed_queue,
          computational_stages=spec.computational_stages,
          gradient_accumulation_count=spec.gradient_accumulation_count,
          repeat_count=self._calc_repeat_count(spec),
          device_mapping=spec.device_mapping,
          accumulate_outfeed=False,
          name="ipu_pipeline_estimator_eval",
          **spec.pipeline_op_kwargs)

    return evaluation_pipeline

  def get_evaluation_loss_and_metrics(self, compiled_evaluation_loop):
    with ops.device(_HOST_DEVICE):
      with ops.control_dependencies([compiled_evaluation_loop]):
        inputs = self._outfeed_queue.dequeue(wait_for_completion=True)

      args, kwargs = loops._body_arguments(inputs)  # pylint: disable=protected-access
      metrics = self._captured_eval_metrics_fn(*args, **kwargs)

    if not isinstance(metrics, dict):
      raise TypeError(("The `eval_metrics_fn` must return a dict, "
                       "but got {}.").format(type(metrics)))

    if model_fn_lib.LOSS_METRIC_KEY not in metrics:
      raise KeyError(
          ("The dict returned from `eval_metrics_fn` "
           "must contain '{}'.").format(model_fn_lib.LOSS_METRIC_KEY))

    loss = metrics.pop(model_fn_lib.LOSS_METRIC_KEY)

    return loss, metrics

  def create_prediction_loop(self):
    def prediction_pipeline():
      spec = self._call_model_fn(model_fn_lib.ModeKeys.PREDICT)

      self._capture_hooks(spec.prediction_hooks)

      return pipelining_ops.pipeline(
          infeed_queue=self._infeed_queue,
          outfeed_queue=self._outfeed_queue,
          computational_stages=spec.computational_stages,
          gradient_accumulation_count=spec.gradient_accumulation_count,
          repeat_count=self._calc_repeat_count(spec),
          device_mapping=spec.device_mapping,
          accumulate_outfeed=False,
          name="ipu_pipeline_estimator_predict",
          **spec.pipeline_op_kwargs)

    return prediction_pipeline

  def get_predictions(self, compiled_prediction_loop):
    with ops.device(_HOST_DEVICE):
      with ops.control_dependencies([compiled_prediction_loop]):
        predictions = self._outfeed_queue.dequeue(wait_for_completion=True)

    if isinstance(predictions, dict):
      return predictions

    assert isinstance(predictions, list)
    if len(predictions) != 1:
      raise ValueError(
          ("The last computational stage must return exactly one prediction "
           "tensor, but got {}").format(len(predictions)))

    return predictions[0]

  def _call_model_fn(self, mode):
    model_fn_args = function_utils.fn_args(self._model_fn)
    kwargs = {}
    if "features" in model_fn_args or "labels" in model_fn_args:
      raise ValueError(
          "The `model_fn` used with `IPUPipelineEstimator` must "
          "not have `features` or `labels` arguments. They are instead "
          "passed to the first pipeline stage function.")
    if "mode" in model_fn_args:
      kwargs["mode"] = mode
    if "params" in model_fn_args:
      kwargs["params"] = self._params
    if "config" in model_fn_args:
      kwargs["config"] = self._config

    estimator_spec = self._model_fn(**kwargs)

    if not isinstance(estimator_spec, IPUPipelineEstimatorSpec):
      raise TypeError("The `model_fn` used with `IPUPipelineEstimator` "
                      "must return `IPUPipelineEstimatorSpec`.")

    if estimator_spec.device_mapping is not None:
      num_devices_required = len(set(estimator_spec.device_mapping))
    else:
      num_devices_required = len(estimator_spec.computational_stages)

    num_shards = self._config.ipu_run_config.num_shards
    if num_devices_required not in range(
        int(num_shards / 2) + 1, num_shards + 1):
      raise ValueError(
          ("This pipeline requires {} devices, but `IPURunConfig.num_shards` "
           "was set to {} (num_shards/2 < pipeline_devices <= num_shards"
           ").").format(num_devices_required, num_shards))

    return estimator_spec


class IPUPipelineEstimator(ipu_estimator._IPUEstimatorBase):  # pylint: disable=protected-access
  """Estimator for pipelining on IPUs.

  `IPUPipelineEstimator`, like :class:`~tensorflow.python.ipu.ipu_estimator.IPUEstimator`,
  handles many of the details of
  running on IPUs, such as placement of operations and tensors, graph
  compilation and usage of data feeds. Additionally, it adds support for
  pipelined execution over multiple IPUs.

  The major API difference from the IPUEstimator is that the provided
  `model_fn` must return a :class:`.IPUPipelineEstimatorSpec`
  that contains the information needed for pipelined execution.

  Data parallelism based on graph replication is supported. Each replica will
  consume `gradient_accumulation_count` batches from the dataset returned by
  the `input_fn` and accumulate the gradients, giving an effective batch size
  of `num_replicas * gradient_accumulation_count * batch_size`. The optimizer
  in the `model_fn` should be wrapped in a
  :class:`~tensorflow.python.ipu.optimizers.CrossReplicaOptimizer`
  in order to average the gradients across the replicas.

  This can further be combined with distributed multi-worker training using the
  :class:`~tensorflow.python.ipu.ipu_multi_worker_strategy.IPUMultiWorkerStrategy`,
  giving a total effective batch size of
  `num_workers * num_replicas * gradient_accumulation_count * batch_size`.

  Refer to the :mod:`~tensorflow.python.ipu.pipelining_ops`
  documentation for more details about pipelining.

  Note: because the `model_fn` is compiled to run on the IPU, you must use the
  `warm_start_from` parameter for a warm start and not the
  `tf.train.init_from_checkpoint` method.


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
                     warm start from, or a `tf.estimator.WarmStartSettings`
                     object to fully configure warm-starting.  If the string
                     filepath is provided instead of a
                     `tf.estimator.WarmStartSettings`, then all variables are
                     warm started, and it is assumed that vocabularies
                     and `tf.Tensor` names are unchanged.
  """
  def __init__(self,
               model_fn,
               model_dir=None,
               config=None,
               params=None,
               warm_start_from=None):
    ipu_device = "/device:IPU:{}".format(config.ipu_run_config.ordinal)

    # pylint: disable=protected-access
    model_function = ipu_estimator._augment_model_fn(model_fn,
                                                     _ModelFnPipelineWrapper,
                                                     ipu_device)
    # pylint: enable=protected-access

    super().__init__(model_fn=model_function,
                     model_dir=model_dir,
                     config=config,
                     params=params,
                     warm_start_from=warm_start_from)

  # Override base validation to allow iterations_per_loop > 1 with
  # distribution, as IPUPipelineEstimator does not have that limitation.
  def _validate_config(self, config):
    pass
