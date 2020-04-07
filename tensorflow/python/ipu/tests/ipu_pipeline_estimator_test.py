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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from tensorflow.python.ipu.ipu_pipeline_estimator import IPUPipelineEstimator
from tensorflow.python.ipu.ipu_pipeline_estimator import IPUPipelineEstimatorSpec
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.summary import summary_iterator
from tensorflow.python.training import gradient_descent


def _make_config(iterations_per_loop=1):
  num_ipus_in_pipeline = 2

  ipu_options = ipu_utils.create_ipu_config()
  ipu_options = ipu_utils.auto_select_ipus(ipu_options,
                                           num_ipus=num_ipus_in_pipeline)
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
                                      pipeline_depth=1,
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
                                      pipeline_depth=1,
                                      device_mapping=[0, 1, 2, 3, 0],
                                      optimizer_function=optimizer_function)

    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([0], [0]))

    ipu_options = ipu_utils.create_ipu_config()
    ipu_options = ipu_utils.auto_select_ipus(ipu_options, num_ipus=2)
    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(
            num_shards=2, iterations_per_loop=1, ipu_options=ipu_options))

    estimator = IPUPipelineEstimator(model_fn=model_fn_with_zero_stages,
                                     config=config)

    with self.assertRaisesRegex(
        ValueError, r"This pipeline requires 4 devices, but "
        "`IPURunConfig.num_shards` was set to 2"):
      estimator.train(input_fn=my_input_fn, steps=1)

  def testMustContainOptimizerFunctionWhenTraining(self):
    def model_fn_without_optimizer_function(mode):
      def stage1(features, labels):
        return features, labels

      def stage2(partial, labels):
        return partial + labels

      return IPUPipelineEstimatorSpec(mode,
                                      computational_stages=[stage1, stage2],
                                      pipeline_depth=1)

    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([0], [0]))

    estimator = IPUPipelineEstimator(
        model_fn=model_fn_without_optimizer_function, config=_make_config())

    with self.assertRaisesRegex(
        ValueError, "must contain `optimizer_function` when training"):
      estimator.train(input_fn=my_input_fn, steps=1)

  @combinations.generate(
      combinations.combine(
          pipeline_depth=[4, 8],
          iterations_per_loop=[1, 2],
      ))
  def testTrainWithAnalyticalGradientReference(self, pipeline_depth,
                                               iterations_per_loop):
    x = 1.5
    y = 1.0
    initial_w = 2.0
    learning_rate = 0.5

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

      return IPUPipelineEstimatorSpec(mode,
                                      computational_stages=[stage1, stage2],
                                      optimizer_function=optimizer_function,
                                      pipeline_depth=pipeline_depth)

    def my_input_fn():
      features = [x] * pipeline_depth * iterations_per_loop
      labels = [y] * pipeline_depth * iterations_per_loop
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
      for _ in range(iterations_per_loop):
        # L(x) = w * x + y
        step_losses.append(expected_w * x + y)
        # dL(x)/dw = x
        # w := w - learning_rate * x
        expected_w -= pipeline_depth * learning_rate * x

      expected_losses.append(np.mean(step_losses))
      self.assertEqual(expected_w, estimator.get_variable_value("w"))

    logged_losses = _get_summary_values(estimator.model_dir,
                                        model_fn_lib.LOSS_METRIC_KEY)
    self.assertEqual(expected_losses, logged_losses)

  @combinations.generate(
      combinations.combine(
          pipeline_depth=[4, 8],
          iterations_per_loop=[1, 2],
      ))
  def testPredictTensor(self, pipeline_depth, iterations_per_loop):
    def my_model_fn(mode):
      self.assertEqual(model_fn_lib.ModeKeys.PREDICT, mode)

      def stage1(features):
        w = variable_scope.get_variable("w", initializer=1)
        partial = w * features
        return partial

      def stage2(partial):
        prediction = partial * partial
        return prediction

      return IPUPipelineEstimatorSpec(mode,
                                      computational_stages=[stage1, stage2],
                                      pipeline_depth=pipeline_depth)

    def my_input_fn():
      features = np.arange(pipeline_depth * iterations_per_loop,
                           dtype=np.int32)
      dataset = dataset_ops.Dataset.from_tensor_slices(features)
      return dataset.batch(1, drop_remainder=True)

    estimator = IPUPipelineEstimator(model_fn=my_model_fn,
                                     config=_make_config(iterations_per_loop))

    num_predictions = pipeline_depth * iterations_per_loop
    predictions = estimator.predict(input_fn=my_input_fn,
                                    num_predictions=num_predictions)

    for i, prediction in enumerate(predictions):
      self.assertEqual(i**2, prediction)

    del predictions  # Release generator resources.

  def testPredictTwoTensorsNotAllowed(self):
    def my_model_fn(mode):
      def stage1(features):
        w = variable_scope.get_variable("w", initializer=1)
        partial = w * features
        return partial

      def stage2(partial):
        prediction = partial * partial
        return prediction, partial

      return IPUPipelineEstimatorSpec(mode,
                                      computational_stages=[stage1, stage2],
                                      pipeline_depth=4)

    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([0]))

    estimator = IPUPipelineEstimator(model_fn=my_model_fn,
                                     config=_make_config())

    with self.assertRaisesRegex(ValueError,
                                "must return exactly one prediction tensor"):
      next(estimator.predict(input_fn=my_input_fn))

  def testPredictDict(self, pipeline_depth=4):
    def my_model_fn(mode):
      def stage1(features):
        w = variable_scope.get_variable("w", initializer=1)
        partial = w * features
        return partial

      def stage2(partial):
        squared = partial * partial
        return {"squared": squared, "partial": partial}

      return IPUPipelineEstimatorSpec(mode,
                                      computational_stages=[stage1, stage2],
                                      pipeline_depth=pipeline_depth)

    def my_input_fn():
      features = np.arange(pipeline_depth, dtype=np.int32)
      dataset = dataset_ops.Dataset.from_tensor_slices(features)
      return dataset.batch(1, drop_remainder=True)

    estimator = IPUPipelineEstimator(model_fn=my_model_fn,
                                     config=_make_config())

    predictions = estimator.predict(input_fn=my_input_fn,
                                    num_predictions=pipeline_depth)

    for i, out in enumerate(predictions):
      self.assertEqual(i, out["partial"])
      self.assertEqual(i**2, out["squared"])

    del predictions  # Release generator resources.

  def testEvaluateMetricsMustBeDict(self):
    def model_fn_without_dict(mode):
      def stage1(features):
        return features * features

      def stage2(partial):
        return partial + partial

      def eval_metrics_fn(partial):
        return partial

      return IPUPipelineEstimatorSpec(mode,
                                      computational_stages=[stage1, stage2],
                                      eval_metrics_fn=eval_metrics_fn,
                                      pipeline_depth=4)

    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([0]))

    estimator = IPUPipelineEstimator(model_fn=model_fn_without_dict,
                                     config=_make_config())

    with self.assertRaisesRegex(TypeError, "must return a dict"):
      estimator.evaluate(input_fn=my_input_fn, steps=1)

  def testEvaluateMetricsMustContainLoss(self):
    def model_fn_without_loss(mode):
      def stage1(features):
        return features * features

      def stage2(partial):
        return partial + partial

      def eval_metrics_fn(prediction):
        return {"prediction": metrics_impl.mean(prediction)}

      return IPUPipelineEstimatorSpec(mode,
                                      computational_stages=[stage1, stage2],
                                      eval_metrics_fn=eval_metrics_fn,
                                      pipeline_depth=4)

    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([0]))

    estimator = IPUPipelineEstimator(model_fn=model_fn_without_loss,
                                     config=_make_config())

    with self.assertRaisesRegex(KeyError, "must contain 'loss'"):
      estimator.evaluate(input_fn=my_input_fn, steps=1)

  def testEvaluateWithStagesMappedToSameIpu(self, pipeline_depth=6):
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
          pipeline_depth=pipeline_depth)

    num_steps = 2
    features = np.arange(pipeline_depth * num_steps, dtype=np.float32)

    def my_input_fn():
      dataset = dataset_ops.Dataset.from_tensor_slices(features)
      return dataset.batch(1, drop_remainder=True)

    estimator = IPUPipelineEstimator(model_fn=my_model_fn,
                                     config=_make_config())

    metrics = estimator.evaluate(input_fn=my_input_fn, steps=num_steps)
    self.assertEqual(np.mean(features), metrics["mean"])
    self.assertEqual(np.mean(np.square(features)), metrics["loss"])


if __name__ == "__main__":
  test.main()
