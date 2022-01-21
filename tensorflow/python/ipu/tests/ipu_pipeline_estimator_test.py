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
# =============================================================================

import glob
import numpy as np

from absl.testing import parameterized
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import ipu_estimator
from tensorflow.python.ipu import ipu_run_config
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.ipu_pipeline_estimator import IPUPipelineEstimator
from tensorflow.python.ipu.ipu_pipeline_estimator import IPUPipelineEstimatorSpec
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer as ga
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.summary import summary_iterator
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util


def _make_config(iterations_per_loop=1):
  num_ipus_in_pipeline = 2

  ipu_options = IPUConfig()
  ipu_options.ipu_model.compile_ipu_code = True
  ipu_options.ipu_model.tiles_per_ipu = 128
  ipu_options.auto_select_ipus = num_ipus_in_pipeline
  return ipu_run_config.RunConfig(ipu_run_config=ipu_run_config.IPURunConfig(
      num_shards=num_ipus_in_pipeline,
      iterations_per_loop=iterations_per_loop,
      ipu_options=ipu_options))


def _get_summary_values(model_dir, tag):
  event_files = glob.glob(model_dir + "/*tfevents*")

  if len(event_files) != 1:
    raise ValueError("Expected exactly one events file in {}, found {}".format(
        model_dir, len(event_files)))

  outputs = []
  for e in summary_iterator.summary_iterator(event_files[0]):
    for v in e.summary.value:
      if v.tag == tag:
        outputs.append(v.simple_value)
  return outputs


class IPUPipelineEstimatorTest(test_util.TensorFlowTestCase,
                               parameterized.TestCase):
  def testModelFnMustNotHaveFeaturesOrLabelsArguments(self):
    def model_fn_with_features_and_labels(features, labels, mode):  # pylint: disable=unused-argument
      pass

    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([0], [0]))

    estimator = IPUPipelineEstimator(
        model_fn=model_fn_with_features_and_labels, config=_make_config())

    with self.assertRaisesRegex(
        ValueError, "must not have `features` or `labels` arguments"):
      estimator.train(input_fn=my_input_fn, steps=1)

  def testModelFnMustReturnIPUPipelineEstimatorSpec(self):
    def model_fn_returning_estimator_spec(mode):
      train_op = control_flow_ops.no_op()
      loss = constant_op.constant(0.0)
      return ipu_estimator.IPUEstimatorSpec(mode, train_op=train_op, loss=loss)

    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([0], [0]))

    estimator = IPUPipelineEstimator(
        model_fn=model_fn_returning_estimator_spec, config=_make_config())

    with self.assertRaisesRegex(TypeError,
                                "must return `IPUPipelineEstimatorSpec`"):
      estimator.train(input_fn=my_input_fn, steps=1)

  def testLengthOfComputationalStagesMustEqualNumShards(self):
    def model_fn_with_zero_stages(mode):
      def optimizer_function():
        pass

      return IPUPipelineEstimatorSpec(mode,
                                      computational_stages=[],
                                      gradient_accumulation_count=1,
                                      optimizer_function=optimizer_function)

    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([0], [0]))

    estimator = IPUPipelineEstimator(model_fn=model_fn_with_zero_stages,
                                     config=_make_config())

    with self.assertRaisesRegex(
        ValueError, "This pipeline requires 0 devices, but "
        "`IPURunConfig.num_shards` was set to 2"):
      estimator.train(input_fn=my_input_fn, steps=1)

  def testNumShardsMustEqualNumUniqueDevices(self):
    def model_fn_with_zero_stages(mode):
      def optimizer_function():
        pass

      return IPUPipelineEstimatorSpec(mode,
                                      computational_stages=[],
                                      gradient_accumulation_count=1,
                                      device_mapping=[0, 1, 2, 3, 0],
                                      optimizer_function=optimizer_function)

    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([0], [0]))

    ipu_options = IPUConfig()
    ipu_options.auto_select_ipus = 2
    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(
            num_shards=2, iterations_per_loop=1, ipu_options=ipu_options))

    estimator = IPUPipelineEstimator(model_fn=model_fn_with_zero_stages,
                                     config=config)

    with self.assertRaisesRegex(
        ValueError, r"This pipeline requires 4 devices, but "
        "`IPURunConfig.num_shards` was set to 2"):
      estimator.train(input_fn=my_input_fn, steps=1)

  def testNumUniqueDevicesBelowNumShardsRange(self):
    def model_fn_with_zero_stages(mode):
      def optimizer_function():
        pass

      return IPUPipelineEstimatorSpec(mode,
                                      computational_stages=[],
                                      gradient_accumulation_count=1,
                                      device_mapping=[0, 1, 0],
                                      optimizer_function=optimizer_function)

    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([0], [0]))

    ipu_options = IPUConfig()
    ipu_options.auto_select_ipus = 4
    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(
            num_shards=4, iterations_per_loop=1, ipu_options=ipu_options))

    estimator = IPUPipelineEstimator(model_fn=model_fn_with_zero_stages,
                                     config=config)

    with self.assertRaisesRegex(
        ValueError, r"This pipeline requires 2 devices, but "
        "`IPURunConfig.num_shards` was set to 4"):
      estimator.train(input_fn=my_input_fn, steps=1)

  def testMustContainOptimizerFunctionWhenTraining(self):
    def model_fn_without_optimizer_function(mode):
      def stage1(features, labels):
        return features, labels

      def stage2(partial, labels):
        return partial + labels

      return IPUPipelineEstimatorSpec(mode,
                                      computational_stages=[stage1, stage2],
                                      gradient_accumulation_count=1)

    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([0], [0]))

    estimator = IPUPipelineEstimator(
        model_fn=model_fn_without_optimizer_function, config=_make_config())

    with self.assertRaisesRegex(
        ValueError, "must contain `optimizer_function` when training"):
      estimator.train(input_fn=my_input_fn, steps=1)

  def testIterationsPerLoopAndGradientAccumulationValidation(self):
    def make_model_fn(gradient_accumulation_count):
      def model_fn(mode):
        def stage1(features, labels):
          return features, labels

        def stage2(partial, labels):
          return partial + labels

        def optimizer_function():
          pass

        return IPUPipelineEstimatorSpec(
            mode,
            computational_stages=[stage1, stage2],
            gradient_accumulation_count=gradient_accumulation_count,
            optimizer_function=optimizer_function)

      return model_fn

    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([0], [0]))

    with self.assertRaisesRegex(
        ValueError, (r"`IPURunConfig.iterations_per_loop` \(got 4\) must be "
                     r"divisible by `gradient_accumulation_count` \(got 3\)")):
      estimator = IPUPipelineEstimator(
          model_fn=make_model_fn(gradient_accumulation_count=3),
          config=_make_config(iterations_per_loop=4))
      estimator.train(input_fn=my_input_fn, steps=4)

    with self.assertRaisesRegex(
        ValueError, (r"`IPURunConfig.iterations_per_loop` \(got 1\) cannot be "
                     r"less than `gradient_accumulation_count` \(got 2\)")):
      estimator = IPUPipelineEstimator(
          model_fn=make_model_fn(gradient_accumulation_count=2),
          config=_make_config(iterations_per_loop=1))
      estimator.train(input_fn=my_input_fn, steps=1)

  @combinations.generate(
      combinations.combine(gradient_accumulation_count=[4, 8],
                           num_weight_updates_per_loop=[1, 2],
                           reduction_method=list(
                               ga.GradientAccumulationReductionMethod)))
  def testTrainWithAnalyticalGradientReference(self,
                                               gradient_accumulation_count,
                                               num_weight_updates_per_loop,
                                               reduction_method):
    x = 1.5
    y = 1.0
    initial_w = 2.0
    learning_rate = 0.5

    iterations_per_loop = (gradient_accumulation_count *
                           num_weight_updates_per_loop)

    def my_model_fn(mode):
      self.assertEqual(model_fn_lib.ModeKeys.TRAIN, mode)

      def stage1(features, labels):
        w = variable_scope.get_variable(name="w", initializer=initial_w)
        partial = w * features
        return partial, labels

      def stage2(partial, labels):
        loss = partial + labels
        return loss

      def optimizer_function(loss):
        opt = gradient_descent.GradientDescentOptimizer(learning_rate)
        return pipelining_ops.OptimizerFunctionOutput(opt, loss)

      return IPUPipelineEstimatorSpec(
          mode,
          computational_stages=[stage1, stage2],
          optimizer_function=optimizer_function,
          gradient_accumulation_count=gradient_accumulation_count,
          reduction_method=reduction_method)

    def my_input_fn():
      num_batches = gradient_accumulation_count * num_weight_updates_per_loop
      features = [x] * num_batches
      labels = [y] * num_batches
      dataset = dataset_ops.Dataset.from_tensor_slices((features, labels))
      return dataset

    estimator = IPUPipelineEstimator(model_fn=my_model_fn,
                                     config=_make_config(iterations_per_loop))

    expected_w = initial_w
    expected_losses = []

    for i in range(2):
      estimator.train(input_fn=my_input_fn, steps=iterations_per_loop)
      self.assertEqual(iterations_per_loop * (i + 1),
                       estimator.get_variable_value("global_step"))

      step_losses = []
      for _ in range(num_weight_updates_per_loop):
        # L(x) = w * x + y
        loss = expected_w * x + y
        step_losses.append(loss)

        # dL(x)/dw = x
        # w := w - learning_rate * x
        if reduction_method == ga.GradientAccumulationReductionMethod.SUM:
          expected_w -= gradient_accumulation_count * learning_rate * x
        else:
          expected_w -= learning_rate * x

      expected_losses.append(np.mean(step_losses))
      self.assertAllClose([expected_w], [estimator.get_variable_value("w")])

    logged_losses = _get_summary_values(estimator.model_dir,
                                        model_fn_lib.LOSS_METRIC_KEY)
    self.assertAllClose(expected_losses, logged_losses)

  @combinations.generate(
      combinations.combine(
          gradient_accumulation_count=[4, 8],
          num_weight_updates_per_loop=[1, 2],
      ))
  def testPredictTensor(self, gradient_accumulation_count,
                        num_weight_updates_per_loop):

    iterations_per_loop = (gradient_accumulation_count *
                           num_weight_updates_per_loop)

    def my_model_fn(mode):
      self.assertEqual(model_fn_lib.ModeKeys.PREDICT, mode)

      def stage1(features):
        w = variable_scope.get_variable("w", initializer=1)
        partial = w * features
        return partial

      def stage2(partial):
        prediction = partial * partial
        return prediction

      return IPUPipelineEstimatorSpec(
          mode,
          computational_stages=[stage1, stage2],
          gradient_accumulation_count=gradient_accumulation_count)

    def my_input_fn():
      features = np.arange(iterations_per_loop, dtype=np.int32)
      dataset = dataset_ops.Dataset.from_tensor_slices(features)
      return dataset.batch(1, drop_remainder=True)

    estimator = IPUPipelineEstimator(model_fn=my_model_fn,
                                     config=_make_config(iterations_per_loop))

    num_predictions = iterations_per_loop
    predictions = estimator.predict(input_fn=my_input_fn,
                                    num_predictions=num_predictions)

    for i, prediction in enumerate(predictions):
      self.assertEqual(i**2, prediction)

    del predictions  # Release generator resources.

  def testPredictTwoTensorsNotAllowed(self):
    gradient_accumulation_count = 4

    def my_model_fn(mode):
      def stage1(features):
        w = variable_scope.get_variable("w", initializer=1)
        partial = w * features
        return partial

      def stage2(partial):
        prediction = partial * partial
        return prediction, partial

      return IPUPipelineEstimatorSpec(
          mode,
          computational_stages=[stage1, stage2],
          gradient_accumulation_count=gradient_accumulation_count)

    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([0]))

    estimator = IPUPipelineEstimator(
        model_fn=my_model_fn,
        config=_make_config(iterations_per_loop=gradient_accumulation_count))

    with self.assertRaisesRegex(ValueError,
                                "must return exactly one prediction tensor"):
      next(estimator.predict(input_fn=my_input_fn))

  def testPredictDict(self, gradient_accumulation_count=4):
    def my_model_fn(mode):
      def stage1(features):
        w = variable_scope.get_variable("w", initializer=1)
        partial = w * features
        return partial

      def stage2(partial):
        squared = partial * partial
        return {"squared": squared, "partial": partial}

      return IPUPipelineEstimatorSpec(
          mode,
          computational_stages=[stage1, stage2],
          gradient_accumulation_count=gradient_accumulation_count)

    def my_input_fn():
      features = np.arange(gradient_accumulation_count, dtype=np.int32)
      dataset = dataset_ops.Dataset.from_tensor_slices(features)
      return dataset.batch(1, drop_remainder=True)

    estimator = IPUPipelineEstimator(
        model_fn=my_model_fn,
        config=_make_config(iterations_per_loop=gradient_accumulation_count))

    predictions = estimator.predict(
        input_fn=my_input_fn, num_predictions=gradient_accumulation_count)

    for i, out in enumerate(predictions):
      self.assertEqual(i, out["partial"])
      self.assertEqual(i**2, out["squared"])

    del predictions  # Release generator resources.

  def testEvaluateMetricsMustBeDict(self):
    gradient_accumulation_count = 4

    def model_fn_without_dict(mode):
      def stage1(features):
        return features * features

      def stage2(partial):
        return partial + partial

      def eval_metrics_fn(partial):
        return partial

      return IPUPipelineEstimatorSpec(
          mode,
          computational_stages=[stage1, stage2],
          eval_metrics_fn=eval_metrics_fn,
          gradient_accumulation_count=gradient_accumulation_count)

    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([0]))

    estimator = IPUPipelineEstimator(
        model_fn=model_fn_without_dict,
        config=_make_config(iterations_per_loop=gradient_accumulation_count))

    with self.assertRaisesRegex(TypeError, "must return a dict"):
      estimator.evaluate(input_fn=my_input_fn,
                         steps=gradient_accumulation_count)

  def testEvaluateMetricsMustContainLoss(self):
    gradient_accumulation_count = 4

    def model_fn_without_loss(mode):
      def stage1(features):
        return features * features

      def stage2(partial):
        return partial + partial

      def eval_metrics_fn(prediction):
        return {"prediction": metrics_impl.mean(prediction)}

      return IPUPipelineEstimatorSpec(
          mode,
          computational_stages=[stage1, stage2],
          eval_metrics_fn=eval_metrics_fn,
          gradient_accumulation_count=gradient_accumulation_count)

    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([0]))

    estimator = IPUPipelineEstimator(
        model_fn=model_fn_without_loss,
        config=_make_config(iterations_per_loop=gradient_accumulation_count))

    with self.assertRaisesRegex(KeyError, "must contain 'loss'"):
      estimator.evaluate(input_fn=my_input_fn,
                         steps=gradient_accumulation_count)

  @combinations.generate(combinations.combine(arg_type=[list, dict]))
  def testEvaluate(self, arg_type):
    num_steps = 2
    gradient_accumulation_count = 2

    def my_model_fn(mode):
      def stage1(features):
        w1 = variable_scope.get_variable("w1", initializer=1.0)
        partial = w1 * features
        return partial

      def stage2(partial):
        squared = partial * partial

        if arg_type is list:
          return [partial, squared]

        assert arg_type is dict
        # Pass in reverse order just to check that they are passed by name.
        return {"squared": squared, "partial": partial}

      def eval_metrics_fn(partial, squared):
        return {
            "mean": metrics_impl.mean(partial),
            "loss": squared,
        }

      return IPUPipelineEstimatorSpec(
          mode,
          computational_stages=[stage1, stage2],
          eval_metrics_fn=eval_metrics_fn,
          gradient_accumulation_count=gradient_accumulation_count)

    features = np.arange(gradient_accumulation_count, dtype=np.float32)

    def my_input_fn():
      dataset = dataset_ops.Dataset.from_tensor_slices(features)
      return dataset.batch(1, drop_remainder=True)

    estimator = IPUPipelineEstimator(
        model_fn=my_model_fn,
        config=_make_config(iterations_per_loop=gradient_accumulation_count))

    metrics = estimator.evaluate(input_fn=my_input_fn, steps=num_steps)
    self.assertEqual(np.mean(features), metrics["mean"])
    self.assertEqual(np.mean(np.square(features)), metrics["loss"])

  def testEvaluateWithStagesMappedToSameIpu(self,
                                            gradient_accumulation_count=3):
    def my_model_fn(mode):
      self.assertEqual(mode, model_fn_lib.ModeKeys.EVAL)

      def stage1(features):
        w1 = variable_scope.get_variable("w1", initializer=1.0)
        partial = w1 * features
        return partial

      def stage2(partial):
        w2 = variable_scope.get_variable("w2", initializer=1.0)
        partial = w2 * partial
        return partial

      def stage3(partial):
        squared = partial * partial
        return partial, squared

      def eval_metrics_fn(partial, squared):
        return {
            "mean": metrics_impl.mean(partial),
            "loss": squared,
        }

      return IPUPipelineEstimatorSpec(
          mode,
          computational_stages=[stage1, stage2, stage3],
          device_mapping=[0, 1, 0],
          eval_metrics_fn=eval_metrics_fn,
          gradient_accumulation_count=gradient_accumulation_count)

    num_steps = 3
    features = np.arange(gradient_accumulation_count, dtype=np.float32)

    def my_input_fn():
      dataset = dataset_ops.Dataset.from_tensor_slices(features)
      return dataset.batch(1, drop_remainder=True)

    estimator = IPUPipelineEstimator(
        model_fn=my_model_fn,
        config=_make_config(iterations_per_loop=gradient_accumulation_count))

    metrics = estimator.evaluate(input_fn=my_input_fn, steps=num_steps)
    self.assertEqual(np.mean(features), metrics["mean"])
    self.assertEqual(np.mean(np.square(features)), metrics["loss"])

  @combinations.generate(
      combinations.combine(num_weight_updates_per_loop=[1, 2]))
  def testPassGlobalStepAsInput(self, num_weight_updates_per_loop):
    x = 1.5
    y = 1.0
    initial_w = 2.0
    gradient_accumulation_count = 4

    iterations_per_loop = (gradient_accumulation_count *
                           num_weight_updates_per_loop)

    def my_model_fn(mode):
      def stage1(global_step, features, labels):
        w = variable_scope.get_variable(name="w", initializer=initial_w)
        partial = w * features
        return global_step, partial, labels

      def stage2(global_step, partial, labels):
        loss = partial + labels
        return global_step, loss

      def optimizer_function(global_step, loss):
        lr = 0.1 - 0.001 * global_step
        opt = gradient_descent.GradientDescentOptimizer(lr)
        return pipelining_ops.OptimizerFunctionOutput(opt, loss)

      def eval_metrics_fn(global_step, loss):
        return {
            "global_step_observed": metrics_impl.mean(global_step),
            "loss": loss,
        }

      global_step_input = math_ops.cast(training_util.get_global_step(),
                                        dtype=np.float32)

      return IPUPipelineEstimatorSpec(
          mode,
          computational_stages=[stage1, stage2],
          optimizer_function=optimizer_function,
          eval_metrics_fn=eval_metrics_fn,
          inputs=[global_step_input],
          gradient_accumulation_count=gradient_accumulation_count)

    def my_input_fn():
      features = [x] * iterations_per_loop
      labels = [y] * iterations_per_loop
      dataset = dataset_ops.Dataset.from_tensor_slices((features, labels))
      return dataset

    estimator = IPUPipelineEstimator(model_fn=my_model_fn,
                                     config=_make_config(iterations_per_loop))

    for i in range(2):
      estimator.train(input_fn=my_input_fn, steps=iterations_per_loop)
      self.assertEqual(iterations_per_loop * (i + 1),
                       estimator.get_variable_value("global_step"))

    out = estimator.evaluate(input_fn=my_input_fn, steps=iterations_per_loop)
    self.assertEqual(2 * iterations_per_loop, out["global_step_observed"])

  def testPassingHooksFromModelFunction(self):
    class _SessionRunCounter(session_run_hook.SessionRunHook):
      def __init__(self):
        self.num_session_runs = 0

      def after_run(self, run_context, run_values):
        self.num_session_runs += 1

    def my_input_fn():
      features = np.array([[1.0]], dtype=np.float32)
      labels = np.array([[2.0]], dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((features, labels))
      return dataset.batch(1, drop_remainder=True).repeat()

    training_hook = _SessionRunCounter()
    evaluation_hook = _SessionRunCounter()
    prediction_hook = _SessionRunCounter()

    def my_model_fn(mode):
      def stage1(features, labels):
        w = variable_scope.get_variable(name="w", initializer=0.1)
        return w * features, labels

      def stage2(partial, labels):
        loss = partial + labels
        return loss

      def optimizer_function(loss):
        opt = gradient_descent.GradientDescentOptimizer(0.5)
        return pipelining_ops.OptimizerFunctionOutput(opt, loss)

      def eval_metrics_fn(loss):
        return {"loss": loss}

      return IPUPipelineEstimatorSpec(mode,
                                      computational_stages=[stage1, stage2],
                                      optimizer_function=optimizer_function,
                                      eval_metrics_fn=eval_metrics_fn,
                                      gradient_accumulation_count=4,
                                      training_hooks=[training_hook],
                                      evaluation_hooks=[evaluation_hook],
                                      prediction_hooks=[prediction_hook])

    estimator = IPUPipelineEstimator(model_fn=my_model_fn,
                                     config=_make_config(4))

    # train
    self.assertEqual(0, training_hook.num_session_runs)
    estimator.train(input_fn=my_input_fn, steps=4)
    self.assertEqual(1, training_hook.num_session_runs)

    # predict: not evaluated before generator is consumed
    self.assertEqual(0, prediction_hook.num_session_runs)
    predictions = estimator.predict(input_fn=my_input_fn)
    self.assertEqual(0, prediction_hook.num_session_runs)
    next(predictions)
    self.assertEqual(1, prediction_hook.num_session_runs)
    del predictions  # Release generator resources

    # evaluate
    self.assertEqual(0, evaluation_hook.num_session_runs)
    estimator.evaluate(my_input_fn, steps=4)
    self.assertEqual(1, evaluation_hook.num_session_runs)


if __name__ == "__main__":
  test.main()
