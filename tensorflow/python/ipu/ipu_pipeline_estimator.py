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
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import function_utils

_HOST_DEVICE = ipu_estimator._HOST_DEVICE  # pylint: disable=protected-access


class IPUPipelineEstimatorSpec(
    collections.namedtuple('IPUPipelineEstimatorSpec', [
        'mode',
        'computational_stages',
        'pipeline_depth',
        'eval_metrics_fn',
        'optimizer_function',
        'device_mapping',
        'pipeline_schedule',
        'offload_weight_update_variables',
        'inputs',
    ])):
  """Ops and objects returned from a `model_fn` and passed to
  :class:`.IPUPipelineEstimator`."""
  def __new__(cls,
              mode,
              computational_stages,
              pipeline_depth,
              eval_metrics_fn=None,
              optimizer_function=None,
              device_mapping=None,
              pipeline_schedule=None,
              offload_weight_update_variables=True,
              inputs=None):
    """Creates a validated `IPUPipelineEstimatorSpec` instance.

    Depending on the value of `mode`, different arguments are required. Namely

    * For `mode == ModeKeys.TRAIN`: the `optimizer_function` is required.
    * For `mode == ModeKeys.EVAL`: the `eval_metrics_fn` is required.

    Refer to the :mod:`~tensorflow.python.ipu.pipelining_ops`
    documentation for more details about pipelining.

    Args:
      mode: A `ModeKeys`. Specifies if this is training, evaluation or
        prediction.
      computational_stages: a list of Python functions, where each function
        represents a computational pipeline stage. The function takes the
        outputs of the previous pipeline state as its inputs.
      pipeline_depth: the number of times each pipeline stage will be executed.
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
      pipeline_schedule: the scheduling algorithm to use for pipeline lowering.
        Must be of type
        :class:`~tensorflow.python.ipu.pipelining_ops.PipelineSchedule`.
      offload_weight_update_variables: If True, any `tf.Variable` which is
        only used by the weight update of the pipeline (for example the
        accumulator variable when using the `tf.MomentumOptimizer`), will be
        stored in the remote memory. During the weight update this variable will
        be streamed onto the device and then streamed back to the remote memory
        after it has been updated. Requires the machine to be configured with
        support for `Poplar remote buffers`. Offloading variables into remote
        memory can reduce maximum memory liveness, but can also increase the
        computation time of the weight update. Note that this option has no
        effect for inference only pipelines.
      inputs: arguments passed to the first pipeline stage. Can be used to pass
        e.g. a learning rate tensor or the `tf.train.get_global_step()` tensor
        that cannot be accessed directly from within a pipeline stage function.

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

    return super().__new__(
        cls,
        mode=mode,
        computational_stages=computational_stages,
        eval_metrics_fn=eval_metrics_fn,
        pipeline_depth=pipeline_depth,
        optimizer_function=optimizer_function,
        device_mapping=device_mapping,
        pipeline_schedule=pipeline_schedule,
        offload_weight_update_variables=offload_weight_update_variables,
        inputs=inputs)


class _ModelFnPipelineWrapper(ipu_estimator._ModelFnWrapperBase):  # pylint: disable=protected-access
  def __init__(self, model_fn, config, params, infeed_queue, outfeed_queue):
    self._model_fn = model_fn
    self._config = config
    self._params = params
    self._infeed_queue = infeed_queue
    self._outfeed_queue = outfeed_queue
    self._captured_eval_metrics_fn = None

  @staticmethod
  def need_outfeed(mode):  # pylint: disable=unused-argument
    return True

  @property
  def captured_hooks(self):
    return []

  def create_training_loop(self):
    def training_pipeline():
      spec = self._call_model_fn(model_fn_lib.ModeKeys.TRAIN)
      return pipelining_ops.pipeline(
          infeed_queue=self._infeed_queue,
          outfeed_queue=self._outfeed_queue,
          computational_stages=spec.computational_stages,
          pipeline_depth=spec.pipeline_depth,
          repeat_count=self._config.ipu_run_config.iterations_per_loop,
          inputs=spec.inputs,
          optimizer_function=spec.optimizer_function,
          device_mapping=spec.device_mapping,
          pipeline_schedule=spec.pipeline_schedule,
          outfeed_loss=True,
          offload_weight_update_variables=spec.offload_weight_update_variables,
          name="ipu_pipeline_estimator_train")

    return training_pipeline

  def get_training_loss_and_op(self, compiled_training_loop):
    with ops.device(_HOST_DEVICE):
      with ops.control_dependencies([compiled_training_loop]):
        loss = self._outfeed_queue.dequeue()

      # Reduce loss over all dimensions (i.e. batch_size, pipeline_depth)
      loss = math_ops.reduce_mean(math_ops.cast(loss, dtypes.float32))

    train_op = compiled_training_loop

    return loss, train_op

  def create_evaluation_loop(self):
    def evaluation_pipeline():
      spec = self._call_model_fn(model_fn_lib.ModeKeys.EVAL)

      assert not self._captured_eval_metrics_fn
      assert spec.eval_metrics_fn
      self._captured_eval_metrics_fn = spec.eval_metrics_fn

      return pipelining_ops.pipeline(
          infeed_queue=self._infeed_queue,
          outfeed_queue=self._outfeed_queue,
          computational_stages=spec.computational_stages,
          pipeline_depth=spec.pipeline_depth,
          repeat_count=self._config.ipu_run_config.iterations_per_loop,
          inputs=spec.inputs,
          device_mapping=spec.device_mapping,
          pipeline_schedule=spec.pipeline_schedule,
          name="ipu_pipeline_estimator_eval")

    return evaluation_pipeline

  def get_evaluation_loss_and_metrics(self, compiled_evaluation_loop):
    with ops.device(_HOST_DEVICE):
      with ops.control_dependencies([compiled_evaluation_loop]):
        inputs = self._outfeed_queue.dequeue()

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
      return pipelining_ops.pipeline(
          infeed_queue=self._infeed_queue,
          outfeed_queue=self._outfeed_queue,
          computational_stages=spec.computational_stages,
          pipeline_depth=spec.pipeline_depth,
          repeat_count=self._config.ipu_run_config.iterations_per_loop,
          inputs=spec.inputs,
          device_mapping=spec.device_mapping,
          pipeline_schedule=spec.pipeline_schedule,
          name="ipu_pipeline_estimator_predict")

    return prediction_pipeline

  def get_predictions(self, compiled_prediction_loop):
    with ops.device(_HOST_DEVICE):
      with ops.control_dependencies([compiled_prediction_loop]):
        predictions = self._outfeed_queue.dequeue()

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
  consume `pipeline_depth` batches from the dataset returned by the `input_fn`
  and accumulate the gradients, giving an effective batch size of
  `num_replicas * pipeline_depth * batch_size`. The optimizer in the `model_fn`
  should be wrapped in an
  :class:`~tensorflow.python.ipu.cross_replica_optimizer.CrossReplicaOptimizer`
  in order to average the gradients across the replicas.

  This can further be combined with distributed multi-worker training using the
  :class:`~tensorflow.python.ipu.ipu_multi_worker_strategy.IPUMultiWorkerStrategy`,
  giving a total effective batch size of
  `num_workers * num_replicas * pipeline_depth * batch_size`.

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
    # pylint: disable=protected-access

    ipu_device = "/device:IPU:{}".format(config.ipu_run_config.ordinal)
    model_function = ipu_estimator._augment_model_fn(model_fn,
                                                     _ModelFnPipelineWrapper,
                                                     ipu_device)
    # pylint: enable=protected-access
    super().__init__(model_fn=model_function,
                     model_dir=model_dir,
                     config=config,
                     params=params,
                     warm_start_from=warm_start_from)
