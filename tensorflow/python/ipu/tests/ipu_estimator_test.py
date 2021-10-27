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
import six
import numpy as np

from absl.testing import parameterized
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.keras import layers
from tensorflow.python import feature_column
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.distribute_config import DistributeConfig
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import ipu_estimator
from tensorflow.python.ipu.ipu_multi_worker_strategy import IPUMultiWorkerStrategyV1
from tensorflow.python.ipu import ipu_run_config
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import googletest
from tensorflow.python.summary import summary_iterator
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import server_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.compat.v1 import disable_v2_behavior


def _dummy_model_fn(features, labels, params):
  _, _, _ = features, labels, params


def _create_regression_dataset(num_samples, num_features):
  np.random.seed(1234)
  target_weights = np.random.rand(num_features, 1).astype(np.float32)
  X = np.random.rand(num_samples, num_features).astype(np.float32)
  y = np.matmul(X, target_weights)
  return X, y


def _is_ipu_estimator(obj):
  return isinstance(obj, ipu_estimator.IPUEstimator)


class _SessionRunCounter(session_run_hook.SessionRunHook):
  def __init__(self):
    self.num_session_runs = 0

  def after_run(self, run_context, run_values):
    self.num_session_runs += 1


class IPUEstimatorTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  def testConstructor(self):
    config = ipu_run_config.RunConfig()
    estimator = ipu_estimator.IPUEstimator(model_fn=_dummy_model_fn,
                                           model_dir="bla",
                                           config=config)
    self.assertTrue(estimator.model_dir == "bla")
    self.assertTrue(isinstance(estimator.config, ipu_run_config.RunConfig))

  @combinations.generate(
      combinations.combine(
          dataset_class=[dataset_ops.DatasetV1, dataset_ops.DatasetV2]))
  def testTrain(self, dataset_class):
    def my_model_fn(features, labels, mode, params):
      self.assertEqual(model_fn_lib.ModeKeys.TRAIN, mode)
      self.assertEqual(2, params["batch_size"])

      with variable_scope.variable_scope("vs", use_resource=True):
        predictions = layers.Dense(units=1)(features)

      loss = losses.mean_squared_error(labels=labels, predictions=predictions)
      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train_op = optimizer.minimize(loss)

      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op)

    def my_input_fn(params):
      dataset = dataset_class.from_tensor_slices(
          _create_regression_dataset(num_samples=1000, num_features=5))
      dataset = dataset.batch(batch_size=params["batch_size"],
                              drop_remainder=True).repeat()
      return dataset

    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(iterations_per_loop=2),
        log_step_count_steps=1)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                           config=config,
                                           train_batch_size=2)

    session_run_counter = _SessionRunCounter()

    estimator.train(input_fn=my_input_fn, steps=4, hooks=[session_run_counter])

    self.assertEqual(2, session_run_counter.num_session_runs)
    self.assertEqual(4, estimator.get_variable_value("global_step"))

    # Calling it again should work
    estimator.train(input_fn=my_input_fn, steps=4, hooks=[session_run_counter])

    self.assertEqual(4, session_run_counter.num_session_runs)
    self.assertEqual(8, estimator.get_variable_value("global_step"))

    # The number of steps is rounded up to the
    # next multiple of `iterations_per_loop`
    estimator.train(input_fn=my_input_fn,
                    max_steps=9,
                    hooks=[session_run_counter])

    self.assertEqual(5, session_run_counter.num_session_runs)
    self.assertEqual(10, estimator.get_variable_value("global_step"))

  def testPassingParams(self):
    exepected_params = {"my_param": 42}

    def my_input_fn(mode, params):
      self.assertEqual(model_fn_lib.ModeKeys.TRAIN, mode)
      self.assertTrue("my_param" in params)
      self.assertTrue(params["my_param"] == 42)
      dataset = tu.create_dual_increasing_dataset(10,
                                                  data_shape=[1],
                                                  label_shape=[1])
      dataset = dataset.batch(batch_size=1, drop_remainder=True)
      return dataset

    def my_model_fn(features, labels, mode, params):
      self.assertEqual(model_fn_lib.ModeKeys.TRAIN, mode)
      self.assertTrue("my_param" in params)
      self.assertTrue(params["my_param"] == 42)
      loss = math_ops.reduce_sum(features + labels, name="loss")
      train_op = array_ops.identity(loss)
      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                           config=ipu_run_config.RunConfig(),
                                           params=exepected_params)

    estimator.train(input_fn=my_input_fn, steps=1)
    self.assertTrue("my_param" in estimator.params)
    self.assertTrue(estimator.params["my_param"] == 42)

  def testHostCallOneArgument(self):
    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([1., 1.], [0., 1.]))

    def my_host_fn(loss):
      loss_sum = variable_scope.get_variable(name="loss_sum", initializer=0.0)
      self.assertEqual("/device:CPU:0", loss_sum.device)
      return loss_sum.assign_add(loss)

    def my_model_fn(features, labels, mode):
      loss = features + labels
      self.assertEqual("/device:IPU:0", loss.device)
      train_op = array_ops.identity(loss)
      host_call = (my_host_fn, [loss])
      return ipu_estimator.IPUEstimatorSpec(mode=mode,
                                            loss=loss,
                                            train_op=train_op,
                                            host_call=host_call)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                           config=ipu_run_config.RunConfig())

    estimator.train(input_fn=my_input_fn, steps=2)
    self.assertEqual(3., estimator.get_variable_value("loss_sum"))

  def testHostCallZeroArguments(self):
    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([1., 1.], [0., 1.]))

    def my_host_fn():
      counter = variable_scope.get_variable(name="host_counter",
                                            initializer=0.0)
      return counter.assign_add(1.0)

    def my_model_fn(features, labels, mode):
      loss = features + labels
      counter = variable_scope.get_variable(name="ipu_counter",
                                            initializer=0.0)
      train_op = counter.assign_add(1.0)
      host_call = (my_host_fn, [])
      return ipu_estimator.IPUEstimatorSpec(mode=mode,
                                            loss=loss,
                                            train_op=train_op,
                                            host_call=host_call)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                           config=ipu_run_config.RunConfig())

    estimator.train(input_fn=my_input_fn, steps=2)
    self.assertEqual(2.0, estimator.get_variable_value("host_counter"))
    self.assertEqual(2.0, estimator.get_variable_value("ipu_counter"))

  def testHostCallNotReturningAnything(self):
    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([1., 1.], [0., 1.]))

    def my_host_fn():
      pass

    def my_model_fn(features, labels, mode):
      loss = features + labels
      train_op = array_ops.identity(loss)
      host_call = (my_host_fn, [])
      return ipu_estimator.IPUEstimatorSpec(mode=mode,
                                            loss=loss,
                                            train_op=train_op,
                                            host_call=host_call)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                           config=ipu_run_config.RunConfig())

    with self.assertRaisesRegex(
        TypeError, "`host_call` return value must be Operation or Tensor"):
      estimator.train(input_fn=my_input_fn, steps=1)

  def testHostCallMultipleIterationsPerLoopNotAllowed(self):
    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([], []))

    def my_host_fn():
      pass

    def my_model_fn(features, labels, mode):
      loss = features + labels
      train_op = array_ops.identity(loss)
      host_call = (my_host_fn, [])
      return ipu_estimator.IPUEstimatorSpec(mode=mode,
                                            loss=loss,
                                            train_op=train_op,
                                            host_call=host_call)

    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(iterations_per_loop=2))

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)
    with self.assertRaisesRegex(
        ValueError, "host_call is not allowed for iterations_per_loop > 1"):
      estimator.train(input_fn=my_input_fn, steps=2)

  def testHostCallTwoArguments(self):
    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([1., 1.], [0., 1.]))

    def my_host_fn(features, labels):
      feature_sum = variable_scope.get_variable(name="feature_sum",
                                                initializer=0.0)
      label_sum = variable_scope.get_variable(name="label_sum",
                                              initializer=0.0)
      return control_flow_ops.group(
          [feature_sum.assign_add(features),
           label_sum.assign_add(labels)])

    def my_model_fn(features, labels, mode):
      loss = features + labels
      self.assertEqual("/device:IPU:0", loss.device)
      train_op = array_ops.identity(loss)
      host_call = (my_host_fn, [features, labels])
      return ipu_estimator.IPUEstimatorSpec(mode=mode,
                                            loss=loss,
                                            train_op=train_op,
                                            host_call=host_call)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                           config=ipu_run_config.RunConfig())

    estimator.train(input_fn=my_input_fn, steps=2)
    self.assertEqual(2., estimator.get_variable_value("feature_sum"))
    self.assertEqual(1., estimator.get_variable_value("label_sum"))

  def testModelFnDoesNotTakeLabels(self):
    def my_input_fn():
      dataset = tu.create_dual_increasing_dataset(10,
                                                  data_shape=[1],
                                                  label_shape=[1])
      dataset = dataset.batch(batch_size=1, drop_remainder=True)
      return dataset

    def my_model_fn(features, mode):
      loss = math_ops.reduce_sum(features, name="loss")
      train_op = array_ops.identity(loss)
      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                           config=ipu_run_config.RunConfig())

    with self.assertRaisesRegex(ValueError, "model_fn does not take labels"):
      estimator.train(input_fn=my_input_fn, steps=1)

  def testNotPassingDataset(self):
    def my_input_fn():
      return 1, 2

    def my_model_fn(features, labels, mode):
      loss = math_ops.reduce_sum(features + labels, name="loss")
      train_op = array_ops.identity(loss)
      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                           config=ipu_run_config.RunConfig())

    with self.assertRaisesRegex(ValueError, "must return Dataset"):
      estimator.train(input_fn=my_input_fn, steps=1)

  def testVerifyTrainFeeds(self):
    def my_model_fn(features, labels, mode):
      self.assertEqual(model_fn_lib.ModeKeys.TRAIN, mode)

      loss = math_ops.reduce_sum(features + labels, name="loss")

      train_op = array_ops.identity(loss)

      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op)

    def my_input_fn():
      dataset = tu.create_dual_increasing_dataset(10,
                                                  data_shape=[1],
                                                  label_shape=[1])
      dataset = dataset.batch(batch_size=1, drop_remainder=True)
      return dataset

    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(iterations_per_loop=1),
        log_step_count_steps=1,
        save_summary_steps=1)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)

    session_run_counter = _SessionRunCounter()

    num_steps = 6
    estimator.train(input_fn=my_input_fn,
                    steps=num_steps,
                    hooks=[session_run_counter])

    self.assertEqual(session_run_counter.num_session_runs, num_steps)

    model_dir = estimator.model_dir
    events_file = glob.glob(model_dir + "/*tfevents*")
    assert len(events_file) == 1
    events_file = events_file[0]
    loss_output = list()
    for e in summary_iterator.summary_iterator(events_file):
      for v in e.summary.value:
        if "loss" in v.tag:
          loss_output.append(v.simple_value)

    self.assertEqual(loss_output, list(np.arange(0.0, 12.0, 2.0)))

  def testTrainWithWarmStart(self):
    class IntrospectionHook(session_run_hook.SessionRunHook):
      def __init__(self):
        self._result = {}
        self._trainable_vars = []

      def begin(self):
        self._trainable_vars = ops.get_collection(
            ops.GraphKeys.TRAINABLE_VARIABLES, scope=".*")
        self._result = {v.name[:-2]: None for v in self._trainable_vars}

      def after_create_session(self, session, coord):
        result = session.run(self._trainable_vars)
        for (k, _), r in zip(six.iteritems(self._result), result):
          self._result[k] = r

      @property
      def result(self):
        return self._result

    def my_model_fn(features, labels, mode):
      self.assertEqual(model_fn_lib.ModeKeys.TRAIN, mode)

      with variable_scope.variable_scope("vs", use_resource=True):
        predictions = layers.Dense(units=1)(features)

      loss = losses.mean_squared_error(labels=labels, predictions=predictions)
      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train_op = optimizer.minimize(loss)

      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op)

    def my_input_fn():
      dataset = dataset_ops.Dataset.from_tensor_slices(
          _create_regression_dataset(num_samples=1000, num_features=5))
      dataset = dataset.batch(batch_size=2, drop_remainder=True).repeat()
      return dataset

    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(iterations_per_loop=2),
        log_step_count_steps=1)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)

    model_dir = estimator.model_dir

    session_run_counter = _SessionRunCounter()

    estimator.train(input_fn=my_input_fn, steps=4, hooks=[session_run_counter])

    variable_names = estimator.get_variable_names()
    estimator_variables = {}
    for n in variable_names:
      if "vs" in n:
        estimator_variables[n] = estimator.get_variable_value(n)

    del estimator
    del session_run_counter
    # Create new estimator which warm starts from previous estimator's output
    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                           config=config,
                                           warm_start_from=model_dir)
    session_run_counter = _SessionRunCounter()
    introspect_hook = IntrospectionHook()

    hooks = [session_run_counter, introspect_hook]
    estimator.train(input_fn=my_input_fn, steps=4, hooks=hooks)

    warm_started_estimator_variables = introspect_hook.result

    self.assertEqual(six.viewkeys(estimator_variables),
                     six.viewkeys(warm_started_estimator_variables))

    for k, _ in six.iteritems(estimator_variables):
      self.assertEqual(estimator_variables[k][0],
                       warm_started_estimator_variables[k][0])

  def testEventDecode(self):
    class EventTraceHook(session_run_hook.SessionRunHook):
      def __init__(self, report_json):
        self._report_json = report_json

      def begin(self):
        self._report_json.create_ipu_event_trace()

      def after_run(self, run_context, run_values):
        self._report_json.parse_log(session=run_context.session)

    def my_model_fn(features, labels, mode):
      self.assertEqual(model_fn_lib.ModeKeys.TRAIN, mode)

      with variable_scope.variable_scope("vs", use_resource=True):
        predictions = layers.Dense(units=1)(features)

      loss = losses.mean_squared_error(labels=labels, predictions=predictions)
      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train_op = optimizer.minimize(loss)

      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op)

    def my_input_fn():
      dataset = dataset_ops.Dataset.from_tensor_slices(
          _create_regression_dataset(num_samples=1000, num_features=5))
      dataset = dataset.batch(batch_size=2, drop_remainder=True).repeat()
      return dataset

    report_json = tu.ReportJSON(self)

    cfg = IPUConfig()
    tu.enable_ipu_events(cfg)
    cfg.auto_select_ipus = 1
    cfg.ipu_model.compile_ipu_code = True

    ipu_config = ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                             ipu_options=cfg)

    run_config = ipu_run_config.RunConfig(ipu_run_config=ipu_config,
                                          log_step_count_steps=1)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                           config=run_config)

    event_trace_hook = EventTraceHook(report_json)
    hooks = [event_trace_hook]
    estimator.train(input_fn=my_input_fn, steps=4, hooks=hooks)

    report_json.assert_num_events(1)

  def testLossAveraging(self):
    def my_model_fn(features, labels, mode):
      self.assertEqual(model_fn_lib.ModeKeys.TRAIN, mode)

      loss = math_ops.reduce_sum(features + labels, name="loss")

      train_op = array_ops.identity(loss)

      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op)

    def my_input_fn():
      dataset = tu.create_dual_increasing_dataset(10,
                                                  data_shape=[1],
                                                  label_shape=[1])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)
      return dataset

    iterations_per_loop = 2
    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(
            iterations_per_loop=iterations_per_loop),
        log_step_count_steps=1,
        save_summary_steps=1)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)

    session_run_counter = _SessionRunCounter()

    num_steps = 6
    num_session_runs = num_steps // iterations_per_loop
    estimator.train(input_fn=my_input_fn,
                    steps=num_steps,
                    hooks=[session_run_counter])

    self.assertEqual(session_run_counter.num_session_runs, num_session_runs)

    model_dir = estimator.model_dir
    events_file = glob.glob(model_dir + "/*tfevents*")
    assert len(events_file) == 1
    events_file = events_file[0]
    loss_output = list()
    for e in summary_iterator.summary_iterator(events_file):
      for v in e.summary.value:
        if "loss" in v.tag:
          loss_output.append(v.simple_value)

    self.assertEqual(loss_output, [6.0, 22.0, 18.0])

  @combinations.generate(
      combinations.combine(estimator_spec_class=[
          model_fn_lib.EstimatorSpec, ipu_estimator.IPUEstimatorSpec
      ]))
  def testEvaluate(self, estimator_spec_class):
    def my_input_fn():
      features = [0., 2.]  # mean: 1
      labels = [1., 3.]  # mean: 2
      return dataset_ops.Dataset.from_tensors((features, labels))

    def my_model_fn(features, labels, mode):
      loss = math_ops.reduce_mean(features + labels)
      eval_metric_ops = {
          "feature_mean": metrics_impl.mean(features),
          "label_mean": metrics_impl.mean(labels),
      }
      return estimator_spec_class(mode,
                                  loss=loss,
                                  eval_metric_ops=eval_metric_ops)

    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(iterations_per_loop=1))
    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)
    scores = estimator.evaluate(my_input_fn, steps=1)
    self.assertEqual(1., scores["feature_mean"])
    self.assertEqual(2., scores["label_mean"])
    self.assertEqual(3., scores[model_fn_lib.LOSS_METRIC_KEY])

  @combinations.generate(
      combinations.combine(estimator_spec_class=[
          model_fn_lib.EstimatorSpec, ipu_estimator.IPUEstimatorSpec
      ]))
  def testEvaluateBinaryClassificationMetrics(self, estimator_spec_class):
    def my_input_fn():
      features = [0, 0, 0, 1]
      labels = [0, 1, 0, 1]
      return dataset_ops.Dataset.from_tensor_slices((features, labels))

    def my_model_fn(features, labels, mode):
      loss = constant_op.constant(0.0)
      eval_metric_ops = {
          "accuracy": metrics_impl.accuracy(labels, features),
          "precision": metrics_impl.precision(labels, features),
          "recall": metrics_impl.recall(labels, features),
      }
      return estimator_spec_class(mode,
                                  loss=loss,
                                  eval_metric_ops=eval_metric_ops)

    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(iterations_per_loop=1))
    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)
    scores = estimator.evaluate(my_input_fn, steps=4)
    self.assertEqual(0.75, scores["accuracy"])
    self.assertEqual(1.0, scores["precision"])
    self.assertEqual(0.5, scores["recall"])

  def testEvaluateMissingEvalMetrics(self):
    def my_input_fn():
      return dataset_ops.Dataset.from_tensors(([], []))

    def my_model_fn(features, labels, mode):
      loss = math_ops.reduce_mean(features + labels)
      return model_fn_lib.EstimatorSpec(mode, loss=loss)

    config = ipu_run_config.RunConfig()
    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)
    with self.assertRaisesRegex(
        ValueError, "must contain either eval_metric_ops or eval_metrics"):
      estimator.evaluate(my_input_fn, steps=1)

  def testEvaluateCannotProvideBothEvalMetrics(self):
    def my_input_fn():
      return dataset_ops.Dataset.from_tensors(([], []))

    def my_model_fn(features, labels, mode):
      eval_metric_ops = {
          "accuracy": metrics_impl.accuracy(labels, features),
      }
      eval_metrics = (lambda: None, [])
      loss = math_ops.reduce_mean(features + labels)
      return ipu_estimator.IPUEstimatorSpec(mode,
                                            loss=loss,
                                            eval_metric_ops=eval_metric_ops,
                                            eval_metrics=eval_metrics)

    config = ipu_run_config.RunConfig()
    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)
    with self.assertRaisesRegex(
        ValueError, "cannot contain both eval_metric_ops and eval_metrics"):
      estimator.evaluate(my_input_fn, steps=1)

  @combinations.generate(combinations.combine(arg_type=[list, dict]))
  def testEvaluateOnHost(self, arg_type):
    def my_input_fn():
      features = [0, 0, 0, 1]
      labels = [0, 1, 0, 1]
      return dataset_ops.Dataset.from_tensor_slices(
          (features, labels)).batch(2, drop_remainder=True)

    def my_metrics_fn(features, labels):
      labels64 = math_ops.cast(labels, np.int64)
      return {
          "accuracy": metrics_impl.accuracy(labels, features),
          "precision": metrics_impl.precision(labels, features),
          "recall": metrics_impl.recall(labels, features),
          "recall_at_1": metrics_impl.recall_at_k(labels64, features, k=1),
          "recall_at_2": metrics_impl.recall_at_k(labels64, features, k=2),
      }

    def my_model_fn(features, labels, mode):
      loss = constant_op.constant(0.0)

      if arg_type is list:
        args = [features, labels]
      else:
        assert arg_type is dict
        # Pass in reverse order just to check that they are passed by name.
        args = {"labels": labels, "features": features}

      eval_metrics = (my_metrics_fn, args)
      return ipu_estimator.IPUEstimatorSpec(mode,
                                            loss=loss,
                                            eval_metrics=eval_metrics)

    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(iterations_per_loop=1))
    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)
    scores = estimator.evaluate(my_input_fn, steps=2)
    self.assertEqual(0.75, scores["accuracy"])
    self.assertEqual(1.0, scores["precision"])
    self.assertEqual(0.5, scores["recall"])
    self.assertEqual(0.5, scores["recall_at_1"])
    self.assertEqual(1.0, scores["recall_at_2"])

  def testEvaluateOnHostWithWrongArgumentType(self):
    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([]))

    def my_metrics_fn():
      pass

    def my_model_fn(features, labels, mode):
      del features, labels
      loss = constant_op.constant(0.0)
      eval_metrics = (my_metrics_fn, loss)
      return ipu_estimator.IPUEstimatorSpec(mode,
                                            loss=loss,
                                            eval_metrics=eval_metrics)

    config = ipu_run_config.RunConfig()
    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)

    with self.assertRaisesRegex(
        TypeError, "second element in `eval_metrics` must be a list or dict"):
      estimator.evaluate(my_input_fn, steps=1)

  @combinations.generate(
      combinations.combine(estimator_class=[
          estimator_lib.Estimator, ipu_estimator.IPUEstimator
      ]))
  def testPredictTensorTwoPerIteration(self, estimator_class):
    def my_input_fn():
      features = [[2.0], [3.0]]
      return dataset_ops.Dataset.from_tensor_slices(features)

    def my_model_fn(features, labels, mode):
      self.assertIsNone(labels)
      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=features,
      )

    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(iterations_per_loop=2))
    estimator = estimator_class(model_fn=my_model_fn, config=config)

    outputs = estimator.predict(input_fn=my_input_fn,
                                yield_single_examples=True)
    self.assertAllEqual(2.0, next(outputs))
    self.assertAllEqual(3.0, next(outputs))
    del outputs  # Release generator resources.

    outputs = estimator.predict(input_fn=my_input_fn,
                                yield_single_examples=False)
    self.assertAllEqual([2.0], next(outputs))
    self.assertAllEqual([3.0], next(outputs))
    del outputs  # Release generator resources.

  @combinations.generate(
      combinations.combine(estimator_class=[
          estimator_lib.Estimator, ipu_estimator.IPUEstimator
      ]))
  def testPredictDictOnePerIteration(self, estimator_class):
    def my_input_fn():
      feature = [2.0]
      return dataset_ops.Dataset.from_tensors(feature)

    def my_model_fn(features, mode):
      return model_fn_lib.EstimatorSpec(
          mode,
          predictions={
              "features": features,
              "features2": features + features,
          },
      )

    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(iterations_per_loop=1))
    estimator = estimator_class(model_fn=my_model_fn, config=config)

    kwargs = {"num_predictions": 1} if _is_ipu_estimator(estimator) else {}

    outputs = list(estimator.predict(input_fn=my_input_fn, **kwargs))
    self.assertEqual(1, len(outputs))
    output = outputs[0]
    self.assertAllEqual(2.0, output["features"])
    self.assertAllEqual(4.0, output["features2"])

    outputs = list(
        estimator.predict(input_fn=my_input_fn,
                          predict_keys=["features"],
                          **kwargs))
    self.assertEqual(1, len(outputs))
    output = outputs[0]
    self.assertAllEqual(2.0, output["features"])
    self.assertFalse("labels" in output)

  @combinations.generate(
      combinations.combine(estimator_class=[
          estimator_lib.Estimator, ipu_estimator.IPUEstimator
      ]))
  def testPredictTensorBatchShouldGiveFlattenedOutput(self, estimator_class):
    def my_input_fn():
      features = [
          [2.0, 3.0],
          [3.0, 2.0],
          [3.0, 2.0],
          [2.0, 3.0],
          [2.0, 3.0],
          [3.0, 2.0],
      ]
      return dataset_ops.Dataset.from_tensor_slices(features).batch(
          2, drop_remainder=True)

    def my_model_fn(features, mode):
      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=math_ops.argmax(features,
                                      axis=-1,
                                      output_type=dtypes.int32),
      )

    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(iterations_per_loop=3))
    estimator = estimator_class(model_fn=my_model_fn, config=config)

    kwargs = {"num_predictions": 6} if _is_ipu_estimator(estimator) else {}
    outputs = estimator.predict(input_fn=my_input_fn,
                                yield_single_examples=True,
                                **kwargs)
    self.assertAllEqual(1, next(outputs))
    self.assertAllEqual(0, next(outputs))
    self.assertAllEqual(0, next(outputs))
    self.assertAllEqual(1, next(outputs))
    self.assertAllEqual(1, next(outputs))
    self.assertAllEqual(0, next(outputs))

    with self.assertRaises(StopIteration):
      next(outputs)

    del outputs  # Release generator resources

    kwargs = {"num_predictions": 3} if _is_ipu_estimator(estimator) else {}
    outputs = estimator.predict(input_fn=my_input_fn,
                                yield_single_examples=False,
                                **kwargs)
    self.assertAllEqual([1, 0], next(outputs))
    self.assertAllEqual([0, 1], next(outputs))
    self.assertAllEqual([1, 0], next(outputs))

    with self.assertRaises(StopIteration):
      next(outputs)

    del outputs  # Release generator resources

  @combinations.generate(
      combinations.combine(estimator_class=[
          estimator_lib.Estimator, ipu_estimator.IPUEstimator
      ]))
  def testPredictDictBatchShouldGiveFlattenedOutput(self, estimator_class):
    def my_input_fn():
      features = [
          [2.0, 3.0],
          [3.0, 2.0],
          [3.0, 2.0],
          [2.0, 3.0],
          [2.0, 3.0],
          [3.0, 2.0],
      ]
      return dataset_ops.Dataset.from_tensor_slices(features).batch(
          2, drop_remainder=True)

    def my_model_fn(features, mode):
      predictions = {
          "predictions":
          math_ops.argmax(features, axis=-1, output_type=dtypes.int32),
      }
      return model_fn_lib.EstimatorSpec(mode, predictions=predictions)

    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(iterations_per_loop=3))
    estimator = estimator_class(model_fn=my_model_fn, config=config)

    kwargs = {"num_predictions": 6} if _is_ipu_estimator(estimator) else {}
    outputs = estimator.predict(input_fn=my_input_fn,
                                yield_single_examples=True,
                                **kwargs)
    self.assertAllEqual(1, next(outputs)["predictions"])
    self.assertAllEqual(0, next(outputs)["predictions"])
    self.assertAllEqual(0, next(outputs)["predictions"])
    self.assertAllEqual(1, next(outputs)["predictions"])
    self.assertAllEqual(1, next(outputs)["predictions"])
    self.assertAllEqual(0, next(outputs)["predictions"])

    with self.assertRaises(StopIteration):
      next(outputs)

    del outputs  # Release generator resources

    kwargs = {"num_predictions": 3} if _is_ipu_estimator(estimator) else {}
    outputs = estimator.predict(input_fn=my_input_fn,
                                yield_single_examples=False,
                                **kwargs)
    self.assertAllEqual([1, 0], next(outputs)["predictions"])
    self.assertAllEqual([0, 1], next(outputs)["predictions"])
    self.assertAllEqual([1, 0], next(outputs)["predictions"])

    with self.assertRaises(StopIteration):
      next(outputs)

    del outputs  # Release generator resources

  def testStepsMustBeMultipleOfIterationsPerLoop(self):
    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(iterations_per_loop=3))
    estimator = ipu_estimator.IPUEstimator(model_fn=_dummy_model_fn,
                                           config=config)

    with self.assertRaisesRegex(ValueError,
                                "must be a multiple of iterations_per_loop"):
      estimator.train(input_fn=lambda: None, steps=1)

    with self.assertRaisesRegex(ValueError,
                                "must be a multiple of iterations_per_loop"):
      estimator.evaluate(input_fn=lambda: None, steps=2)

  # Since IPUModel can't use auto_select_ipus
  @tu.test_uses_ipus(4, allow_ipu_model=False)
  def testIPURunConfig(self):
    with self.assertRaisesRegex(ValueError,
                                "configuration requires more than one device"):
      ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                  num_replicas=3,
                                  ipu_options=None)

    with self.assertRaisesRegex(ValueError,
                                "configuration requires more than one device"):
      ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                  num_shards=2,
                                  ipu_options=None)

    ipu_config = ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                             num_replicas=1,
                                             ipu_options=None)
    self.assertIsInstance(ipu_config, ipu_run_config.IPURunConfig)

    ipu_config = ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                             num_shards=1,
                                             ipu_options=None)
    self.assertIsInstance(ipu_config, ipu_run_config.IPURunConfig)

    ipu_options = IPUConfig()
    tu.enable_ipu_events(ipu_options)
    ipu_options.auto_select_ipus = 1
    ipu_config = ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                             ipu_options=ipu_options)
    self.assertIsInstance(ipu_config, ipu_run_config.IPURunConfig)

    with self.assertRaisesRegex(ValueError, "`IpuOptions` configured with"):
      ipu_options = IPUConfig()
      tu.enable_ipu_events(ipu_options)
      ipu_options.auto_select_ipus = 3
      ipu_config = ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                               ipu_options=ipu_options)

    with self.assertRaisesRegex(
        ValueError, r"`IPURunConfig` configured with 4 devices "
        r"\(4 replicas times 1 shards\), "
        r"but `IpuOptions` configured with 3 devices"):
      ipu_options = IPUConfig()
      tu.enable_ipu_events(ipu_options)
      ipu_options.auto_select_ipus = 3
      ipu_config = ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                               num_replicas=4,
                                               ipu_options=ipu_options)

    with self.assertRaisesRegex(
        ValueError, r"`IPURunConfig` configured with 4 devices "
        r"\(4 replicas times 1 shards\), "
        r"but `IpuOptions` configured with 1 devices"):
      ipu_options = IPUConfig()
      tu.enable_ipu_events(ipu_options)
      ipu_options.auto_select_ipus = [4, 1]
      ipu_config = ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                               num_replicas=4,
                                               ipu_options=ipu_options,
                                               ordinal=1)

    ipu_options = IPUConfig()
    tu.enable_ipu_events(ipu_options)
    ipu_options.auto_select_ipus = [2, 4]
    ipu_config = ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                             num_replicas=4,
                                             ipu_options=ipu_options,
                                             ordinal=1)
    self.assertIsInstance(ipu_config, ipu_run_config.IPURunConfig)

    ipu_options = IPUConfig()
    tu.enable_ipu_events(ipu_options)
    ipu_options.auto_select_ipus = 4
    ipu_config = ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                             num_replicas=4,
                                             ipu_options=ipu_options)
    self.assertIsInstance(ipu_config, ipu_run_config.IPURunConfig)

    ipu_options = IPUConfig()
    tu.enable_ipu_events(ipu_options)
    ipu_options.auto_select_ipus = 4
    ipu_config = ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                             num_replicas=2,
                                             num_shards=2,
                                             ipu_options=ipu_options)
    self.assertIsInstance(ipu_config, ipu_run_config.IPURunConfig)

    ipu_options = IPUConfig()
    tu.enable_ipu_events(ipu_options)
    ipu_options.select_ipus = [0, 1, 2, 3]
    ipu_config = ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                             num_shards=4,
                                             ipu_options=ipu_options)

    self.assertIsInstance(ipu_config, ipu_run_config.IPURunConfig)

    ipu_options = IPUConfig()
    tu.enable_ipu_events(ipu_options)
    ipu_options.select_ipus = [0, 1, 2, 3]
    ipu_config = ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                             num_replicas=2,
                                             num_shards=2,
                                             ipu_options=ipu_options)

    self.assertIsInstance(ipu_config, ipu_run_config.IPURunConfig)

    ipu_options = IPUConfig()
    tu.enable_ipu_events(ipu_options)
    ipu_options.select_ipus = [0]
    ipu_config = ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                             num_replicas=1,
                                             num_shards=1,
                                             ipu_options=ipu_options)

    self.assertIsInstance(ipu_config, ipu_run_config.IPURunConfig)

  @combinations.generate(
      combinations.combine(estimator_class=[
          estimator_lib.Estimator, ipu_estimator.IPUEstimator
      ]))
  def testDatasetWithDicts(self, estimator_class):
    def my_input_fn():
      features = {
          "x0": np.array([[1.0]], dtype=np.float32),
          "x1": np.array([[2.0]], dtype=np.float32)
      }
      labels = {
          "y0": np.array([[3.0]], dtype=np.float32),
          "y1": np.array([[4.0]], dtype=np.float32)
      }
      dataset = dataset_ops.Dataset.from_tensor_slices((features, labels))
      return dataset.batch(1, drop_remainder=True)

    def my_model_fn(features, labels, mode):
      if mode == model_fn_lib.ModeKeys.PREDICT:
        self.assertIsNone(labels)
        loss = math_ops.reduce_sum(features["x0"] + features["x1"], axis=-1)
      else:
        loss = math_ops.reduce_sum(features["x0"] + features["x1"] +
                                   labels["y0"] + labels["y1"],
                                   axis=-1)

      train_op = array_ops.identity(loss)
      predictions = loss
      eval_metric_ops = {"mean_loss": metrics_impl.mean(loss)}
      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op,
                                        predictions=predictions,
                                        eval_metric_ops=eval_metric_ops)

    estimator = estimator_class(model_fn=my_model_fn,
                                config=ipu_run_config.RunConfig())

    # train
    estimator.train(input_fn=my_input_fn, steps=1)

    # predict
    predictions = estimator.predict(input_fn=my_input_fn,
                                    yield_single_examples=True)
    self.assertEqual(3.0, next(predictions))
    del predictions  # Release generator resources

    # evaluate
    scores = estimator.evaluate(my_input_fn, steps=1)
    self.assertAllClose(10.0, scores["loss"])

  @combinations.generate(
      combinations.combine(estimator_class=[
          estimator_lib.Estimator, ipu_estimator.IPUEstimator
      ]))
  def testFlatDatasetWithDict(self, estimator_class):
    def my_input_fn():
      dataset = dataset_ops.Dataset.from_tensor_slices({
          "input_ids": [[0], [1]],
          "input_mask": [[2], [3]],
          "labels": [[4], [5]],
      })
      return dataset.batch(1, drop_remainder=True)

    def my_model_fn(features, mode):
      loss = math_ops.reduce_sum(features["input_ids"] +
                                 features["input_mask"] + features["labels"],
                                 axis=-1)
      train_op = array_ops.identity(loss)
      predictions = loss
      eval_metric_ops = {"mean_loss": metrics_impl.mean(loss)}
      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op,
                                        predictions=predictions,
                                        eval_metric_ops=eval_metric_ops)

    estimator = estimator_class(model_fn=my_model_fn,
                                config=ipu_run_config.RunConfig())

    # train
    estimator.train(input_fn=my_input_fn, steps=2)

    # predict
    predictions = estimator.predict(input_fn=my_input_fn)
    self.assertAllEqual(6.0, next(predictions))
    self.assertAllEqual(9.0, next(predictions))
    del predictions  # Release generator resources

    # evaluate
    scores = estimator.evaluate(my_input_fn, steps=2)
    self.assertAllEqual(7.5, scores["loss"])

  @combinations.generate(
      combinations.combine(estimator_class=[
          estimator_lib.Estimator, ipu_estimator.IPUEstimator
      ]))
  def testTupleDatasetWithWrongNumberOfElements(self, estimator_class):
    def input_fn_with_one_element_tuple():
      dataset = dataset_ops.Dataset.from_tensor_slices(([[0], [1]],))
      return dataset.batch(1, drop_remainder=True)

    def input_fn_with_three_element_tuple():
      dataset = dataset_ops.Dataset.from_tensor_slices((
          [[0], [1]],
          [[2], [3]],
          [[4], [5]],
      ))
      return dataset.batch(1, drop_remainder=True)

    def my_model_fn(features, labels, mode):
      loss = math_ops.reduce_sum(features + labels, axis=-1)
      train_op = array_ops.identity(loss)
      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op)

    estimator = estimator_class(model_fn=my_model_fn,
                                config=ipu_run_config.RunConfig())

    with self.assertRaisesRegex(
        ValueError,
        r"input_fn should return \(features, labels\) as a len 2 tuple."):
      estimator.train(input_fn=input_fn_with_one_element_tuple, steps=1)

    with self.assertRaisesRegex(
        ValueError,
        r"input_fn should return \(features, labels\) as a len 2 tuple."):
      estimator.train(input_fn=input_fn_with_three_element_tuple, steps=1)

  @combinations.generate(
      combinations.combine(estimator_class=[
          estimator_lib.Estimator, ipu_estimator.IPUEstimator
      ]))
  def testPassingHooksFromModelFunction(self, estimator_class):
    def my_input_fn():
      features = np.array([[1.0]], dtype=np.float32)
      labels = np.array([[2.0]], dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((features, labels))
      return dataset.batch(1, drop_remainder=True)

    training_hook = _SessionRunCounter()
    evaluation_hook = _SessionRunCounter()
    prediction_hook = _SessionRunCounter()

    def my_model_fn(features, labels, mode):
      if mode == model_fn_lib.ModeKeys.PREDICT:
        self.assertIsNone(labels)
        loss = features
      else:
        loss = features + labels

      train_op = array_ops.identity(loss)
      predictions = loss
      eval_metric_ops = {"mean_loss": metrics_impl.mean(loss)}
      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op,
                                        predictions=predictions,
                                        eval_metric_ops=eval_metric_ops,
                                        training_hooks=[training_hook],
                                        evaluation_hooks=[evaluation_hook],
                                        prediction_hooks=[prediction_hook])

    estimator = estimator_class(model_fn=my_model_fn,
                                config=ipu_run_config.RunConfig())

    # train
    self.assertEqual(0, training_hook.num_session_runs)
    estimator.train(input_fn=my_input_fn, steps=1)
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
    estimator.evaluate(my_input_fn, steps=1)
    self.assertEqual(1, evaluation_hook.num_session_runs)

  @combinations.generate(combinations.combine(iterations_per_loop=[1, 2]))
  def testIncrementingGlobalStepInModelFunctionShouldRaiseError(
      self, iterations_per_loop):
    def my_model_fn(features, labels, mode):
      predictions = layers.Dense(units=1)(features)
      loss = losses.mean_squared_error(labels=labels, predictions=predictions)
      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train_op = optimizer.minimize(
          loss, global_step=training_util.get_global_step())
      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op)

    def my_input_fn():
      dataset = dataset_ops.Dataset.from_tensor_slices(
          _create_regression_dataset(num_samples=2, num_features=5))
      dataset = dataset.batch(batch_size=1, drop_remainder=True).repeat()
      return dataset

    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(
            iterations_per_loop=iterations_per_loop))
    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)

    with self.assertRaisesRegex(
        ValueError,
        "Illegal increment of the `global_step` variable in the `model_fn`"):
      estimator.train(input_fn=my_input_fn, steps=iterations_per_loop)

  @combinations.generate(combinations.combine(iterations_per_loop=[1, 2]))
  def testReadGlobalStepInModelFunction(self, iterations_per_loop):
    def my_model_fn(features, mode):
      loss = math_ops.reduce_sum(features)
      global_step = training_util.get_global_step()
      eval_metric_ops = {"mean_global_step": metrics_impl.mean(global_step)}
      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        eval_metric_ops=eval_metric_ops)

    def my_input_fn():
      features = [0, 0, 0, 0]
      return dataset_ops.Dataset.from_tensor_slices(features)

    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(
            iterations_per_loop=iterations_per_loop))
    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)

    num_steps = 4
    metrics = estimator.evaluate(input_fn=my_input_fn, steps=num_steps)

    # The global step is cached and only incremented on the host
    # after the loop has completed. So the resolution of the observed
    # global step will be `iterations_per_loop`.
    sample_period = iterations_per_loop
    sampled = [(i // sample_period) * sample_period for i in range(num_steps)]
    self.assertEqual(np.mean(sampled), metrics["mean_global_step"])

  def testPredictWithFeatureColumns(self):
    def my_model_fn(features, mode):
      return model_fn_lib.EstimatorSpec(mode=mode, predictions=features)

    def my_input_fn():
      # Add some feature columns.
      feature_columns = []

      age = feature_column.numeric_column("age")
      feature_columns.append(age)

      age_buckets = feature_column.bucketized_column(age,
                                                     boundaries=[20, 40, 60])
      feature_columns.append(age_buckets)

      direction = feature_column.categorical_column_with_vocabulary_list(
          "direction", ["north", "east", "south", "west"])

      embedding = feature_column.embedding_column(
          direction, dimension=4, initializer=init_ops.Identity())
      feature_columns.append(embedding)

      dataset = dataset_ops.Dataset.from_tensor_slices({
          "age": [[18], [25], [55], [90]],
          "direction": [["west"], ["south"], ["east"], ["north"]]
      })

      # Add a dataset mapping to dense features.
      feature_layer = layers.DenseFeatures(feature_columns)
      dataset = dataset.map(feature_layer)
      dataset = dataset.batch(2, drop_remainder=True)
      return dataset

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                           config=ipu_run_config.RunConfig())

    outputs = estimator.predict(my_input_fn)

    self.assertAllEqual([[18., 1., 0., 0., 0., 0., 0., 0., 1.]], next(outputs))
    self.assertAllEqual([[25., 0., 1., 0., 0., 0., 0., 1., 0.]], next(outputs))
    self.assertAllEqual([[55., 0., 0., 1., 0., 0., 1., 0., 0.]], next(outputs))
    self.assertAllEqual([[90., 0., 0., 0., 1., 1., 0., 0., 0.]], next(outputs))

    del outputs  # Release generator resources.

  def testBatchSizeCalculationWithReplication(self):
    def input_fn_without_params():
      pass

    class ParamsReceived(RuntimeError):
      def __init__(self, params):
        super().__init__(str(params))
        self.params = params

    def mock_input_fn(params):
      raise ParamsReceived(params)

    # Testing with replication here is fine with the IPU model since we never
    # attempt to compile anything.
    ipu_options = IPUConfig()
    ipu_options.auto_select_ipus = 2
    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(num_replicas=2,
                                                   ipu_options=ipu_options))

    for param in ["train_batch_size", "eval_batch_size", "predict_batch_size"]:
      with self.assertRaisesRegex(
          ValueError,
          "batch_size cannot be passed in params when a batch size argument "
          "is passed"):
        ipu_estimator.IPUEstimator(model_fn=_dummy_model_fn,
                                   config=config,
                                   **{param: 1},
                                   params={"batch_size": 1})

      with self.assertRaisesRegex(
          ValueError, r"{} \(got 0\) must be positive".format(param)):
        ipu_estimator.IPUEstimator(model_fn=_dummy_model_fn,
                                   config=config,
                                   **{param: 0})

      with self.assertRaisesRegex(
          ValueError, r"{} \(got 1\) must be divisible by "
          r"num_workers \* num_replicas \(1 \* 2\)".format(param)):
        ipu_estimator.IPUEstimator(model_fn=_dummy_model_fn,
                                   config=config,
                                   **{param: 1})

    estimator = ipu_estimator.IPUEstimator(model_fn=_dummy_model_fn,
                                           config=config,
                                           train_batch_size=2,
                                           eval_batch_size=4,
                                           predict_batch_size=6)

    with self.assertRaisesRegex(
        ValueError,
        r"input_fn must have params argument to receive params\['batch_size'\]"
    ):
      estimator.train(input_fn_without_params, steps=1)

    with self.assertRaisesRegex(ParamsReceived, "'batch_size': 1"):
      estimator.train(mock_input_fn, steps=1)

    with self.assertRaisesRegex(ParamsReceived, "'batch_size': 2"):
      estimator.evaluate(mock_input_fn, steps=1)

    with self.assertRaisesRegex(ParamsReceived, "'batch_size': 3"):
      next(estimator.predict(mock_input_fn))

    # Test that only passing train_batch_size does not pass
    # params["batch_size"] during evaluate and predict.
    estimator = ipu_estimator.IPUEstimator(model_fn=_dummy_model_fn,
                                           config=config,
                                           train_batch_size=2)

    with self.assertRaisesRegex(ParamsReceived, "'batch_size': 1"):
      estimator.train(mock_input_fn, steps=1)

    with self.assertRaises(ParamsReceived) as raised:
      estimator.evaluate(mock_input_fn, steps=1)
    self.assertNotIn("batch_size", raised.exception.params)

    with self.assertRaises(ParamsReceived) as raised:
      next(estimator.predict(mock_input_fn))
    self.assertNotIn("batch_size", raised.exception.params)

    # Test that passing params["batch_size"] manually is fine as long
    # as none of the batch size arguments are passed.
    estimator = ipu_estimator.IPUEstimator(model_fn=_dummy_model_fn,
                                           config=config,
                                           params={"batch_size": 42})

    with self.assertRaisesRegex(ParamsReceived, "'batch_size': 42"):
      estimator.train(mock_input_fn, steps=1)

    with self.assertRaisesRegex(ParamsReceived, "'batch_size': 42"):
      estimator.evaluate(mock_input_fn, steps=1)

    with self.assertRaisesRegex(ParamsReceived, "'batch_size': 42"):
      next(estimator.predict(mock_input_fn))

  def testInvalidPrefetchDepth(self):
    def my_model_fn(features, labels, mode):
      del features, labels, mode

    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(([1.], [0.]))

    estimator = ipu_estimator.IPUEstimator(
        model_fn=my_model_fn,
        config=ipu_run_config.RunConfig(
            ipu_run_config=ipu_run_config.IPURunConfig(prefetch_depth=-1)))

    with self.assertRaisesRegex(ValueError,
                                "prefetch_depth must be greater than zero"):
      estimator.train(my_input_fn, steps=1)


class IPUEstimatorWithStrategyTest(multi_worker_test_base.MultiWorkerTestBase):
  """Tests using multiple threads in the same processes."""
  @classmethod
  def setUpClass(cls):  # pylint: disable=arguments-differ
    cls._num_workers = 2
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=cls._num_workers, num_ps=0, has_chief=False)

  def setUp(self):
    disable_v2_behavior()
    # We use a different key_base for each test so that collective keys won't be
    # reused.
    IPUMultiWorkerStrategyV1._collective_key_base += 100000
    super().setUp()

  def testDistributedMultipleIterationsPerLoopNotImplemented(self):

    cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=2, num_ps=0, has_chief=False)

    def function(task_type, task_id, _num_gpus):
      del _num_gpus
      train_cluster = multi_worker_util.normalize_cluster_spec(cluster_spec)
      strategy = IPUMultiWorkerStrategyV1(
          SimpleClusterResolver(train_cluster,
                                task_type=task_type,
                                task_id=task_id))

      # Setting config.train_distribute
      config = ipu_run_config.RunConfig(
          ipu_run_config=ipu_run_config.IPURunConfig(iterations_per_loop=2),
          train_distribute=strategy)
      with self.assertRaisesRegex(
          NotImplementedError,
          r"iterations_per_loop > 1 \(got 2\) not supported with distribution"
      ):
        ipu_estimator.IPUEstimator(model_fn=_dummy_model_fn, config=config)

      # Setting config.experimental_distribute.train_distribute
      config = ipu_run_config.RunConfig(
          ipu_run_config=ipu_run_config.IPURunConfig(iterations_per_loop=2),
          experimental_distribute=DistributeConfig(train_distribute=strategy))
      with self.assertRaisesRegex(
          NotImplementedError,
          r"iterations_per_loop > 1 \(got 2\) not supported with distribution"
      ):
        ipu_estimator.IPUEstimator(model_fn=_dummy_model_fn, config=config)

    self._run_between_graph_clients(function, cluster_spec, num_gpus=0)

  def testBatchSizeCalculationWithReplicationAndDistribution(self):

    cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=3, num_ps=0, has_chief=False)

    class ParamsReceived(RuntimeError):
      pass

    def mock_input_fn(params):
      raise ParamsReceived(str(params))

    def function(task_type, task_id, _num_gpus):
      del _num_gpus
      train_spec = {"worker": cluster_spec["worker"][:2]}
      train_cluster = multi_worker_util.normalize_cluster_spec(train_spec)
      eval_cluster = multi_worker_util.normalize_cluster_spec(cluster_spec)

      train_distribute = None
      if task_id < 2:
        train_distribute = IPUMultiWorkerStrategyV1(
            SimpleClusterResolver(train_cluster,
                                  task_type=task_type,
                                  task_id=task_id))

      eval_distribute = IPUMultiWorkerStrategyV1(
          SimpleClusterResolver(eval_cluster,
                                task_type=task_type,
                                task_id=task_id))

      # Testing with replication here is fine with the IPU model since we never
      # attempt to compile anything.
      ipu_options = IPUConfig()
      ipu_options.auto_select_ipus = 2
      config = ipu_run_config.RunConfig(
          ipu_run_config=ipu_run_config.IPURunConfig(num_replicas=2,
                                                     ipu_options=ipu_options),
          train_distribute=train_distribute,
          eval_distribute=eval_distribute)

      if task_id < 2:
        with self.assertRaisesRegex(
            ValueError, r"train_batch_size \(got 1\) must be divisible by "
            r"num_workers \* num_replicas \(2 \* 2\)"):
          ipu_estimator.IPUEstimator(model_fn=_dummy_model_fn,
                                     config=config,
                                     train_batch_size=1)

      with self.assertRaisesRegex(
          ValueError, r"eval_batch_size \(got 1\) must be divisible by "
          r"num_workers \* num_replicas \(3 \* 2\)"):
        ipu_estimator.IPUEstimator(model_fn=_dummy_model_fn,
                                   config=config,
                                   eval_batch_size=1)

      estimator = ipu_estimator.IPUEstimator(model_fn=_dummy_model_fn,
                                             config=config,
                                             train_batch_size=4,
                                             eval_batch_size=12,
                                             predict_batch_size=6)

      if task_id < 2:
        with self.assertRaisesRegex(ParamsReceived, "'batch_size': 1"):
          estimator.train(mock_input_fn, steps=1)

      with self.assertRaisesRegex(ParamsReceived, "'batch_size': 2"):
        estimator.evaluate(mock_input_fn, steps=1)

      with self.assertRaisesRegex(ParamsReceived, "'batch_size': 3"):
        next(estimator.predict(mock_input_fn))

    self._run_between_graph_clients(function, cluster_spec, num_gpus=0)


if __name__ == "__main__":
  googletest.main()
