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
import six
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import session_run_hook
from tensorflow.python.summary import summary_iterator
from tensorflow.keras import layers
from tensorflow.python.ipu import ipu_estimator
from tensorflow.python.ipu import ipu_run_config
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu


def _dummy_model_fn(features, labels, params):
  _, _, _ = features, labels, params


def _create_regression_dataset(num_samples, num_features):
  np.random.seed(1234)
  target_weights = np.random.rand(num_features, 1).astype(np.float32)
  X = np.random.rand(num_samples, num_features).astype(np.float32)
  y = np.matmul(X, target_weights)
  return X, y


class _SessionRunCounter(session_run_hook.SessionRunHook):
  def __init__(self):
    self.num_session_runs = 0

  def after_run(self, run_context, run_values):
    self.num_session_runs += 1


class IPUEstimatorTest(test_util.TensorFlowTestCase):
  def testConstructor(self):
    config = ipu_run_config.RunConfig()
    estimator = ipu_estimator.IPUEstimator(model_fn=_dummy_model_fn,
                                           model_dir="bla",
                                           config=config)
    self.assertTrue(estimator.model_dir == "bla")
    self.assertTrue(isinstance(estimator.config, ipu_run_config.RunConfig))

  def testMoreThanOneIPUNotImplemented(self):
    ipu_options = ipu_utils.create_ipu_config()
    ipu_utils.select_ipus(ipu_options, indices=[0, 1])
    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(ipu_options=ipu_options))
    with self.assertRaisesRegexp(NotImplementedError, "Only one IPU"):
      estimator = ipu_estimator.IPUEstimator(model_fn=_dummy_model_fn,
                                             config=config)

  def testMoreThanOneAutoSelectedIPUNotImplemented(self):
    ipu_options = ipu_utils.create_ipu_config()
    ipu_utils.auto_select_ipus(ipu_options, num_ipus=2)
    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(ipu_options=ipu_options))
    with self.assertRaisesRegexp(NotImplementedError, "Only one IPU"):
      estimator = ipu_estimator.IPUEstimator(model_fn=_dummy_model_fn,
                                             config=config)

  def testPredictAndEvaluateNotImplemented(self):
    config = ipu_run_config.RunConfig()
    estimator = ipu_estimator.IPUEstimator(model_fn=_dummy_model_fn,
                                           config=config)
    with self.assertRaises(NotImplementedError):
      estimator.predict(input_fn=lambda: ())
    with self.assertRaises(NotImplementedError):
      estimator.evaluate(input_fn=lambda: ())

  def testTrain(self):
    def my_model_fn(features, labels, mode):
      self.assertEquals(model_fn_lib.ModeKeys.TRAIN, mode)

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

    session_run_counter = _SessionRunCounter()

    estimator.train(input_fn=my_input_fn, steps=4, hooks=[session_run_counter])

    self.assertEquals(2, session_run_counter.num_session_runs)
    self.assertEquals(4, estimator.get_variable_value("global_step"))

    # Calling it again should work
    estimator.train(input_fn=my_input_fn, steps=4, hooks=[session_run_counter])

    self.assertEquals(4, session_run_counter.num_session_runs)
    self.assertEquals(8, estimator.get_variable_value("global_step"))

    # The number of steps is rounded up to the
    # next multiple of `iterations_per_loop`
    estimator.train(input_fn=my_input_fn,
                    max_steps=9,
                    hooks=[session_run_counter])

    self.assertEquals(5, session_run_counter.num_session_runs)
    self.assertEquals(10, estimator.get_variable_value("global_step"))

  def testPassingParams(self):
    exepected_params = ("my_param", 42)

    def my_input_fn(mode, params, config):
      self.assertEquals(model_fn_lib.ModeKeys.TRAIN, mode)
      self.assertEquals(exepected_params, params)
      dataset = tu.create_dual_increasing_dataset(10,
                                                  data_shape=[1],
                                                  label_shape=[1])
      dataset = dataset.batch(batch_size=1, drop_remainder=True)
      return dataset

    def my_model_fn(features, labels, mode, params):
      self.assertEquals(model_fn_lib.ModeKeys.TRAIN, mode)
      self.assertEquals(exepected_params, params)
      loss = math_ops.reduce_sum(features + labels, name="loss")
      train_op = array_ops.identity(loss)
      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                           config=ipu_run_config.RunConfig(),
                                           params=exepected_params)

    estimator.train(input_fn=my_input_fn, steps=1)

  def testModelFnDoesNotTakeLabels(self):
    def my_input_fn():
      dataset = tu.create_dual_increasing_dataset(10,
                                                  data_shape=[1],
                                                  label_shape=[1])
      dataset = dataset.batch(batch_size=1, drop_remainder=True)
      return dataset

    def my_model_fn(features):
      loss = math_ops.reduce_sum(features, name="loss")
      train_op = array_ops.identity(loss)
      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                           config=ipu_run_config.RunConfig())

    with self.assertRaisesRegexp(ValueError, "model_fn does not take labels"):
      estimator.train(input_fn=my_input_fn, steps=1)

  def testNotPassingDataset(self):
    def my_input_fn():
      return 1, 2

    def my_model_fn(features, labels):
      loss = math_ops.reduce_sum(features + labels, name="loss")
      train_op = array_ops.identity(loss)
      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                           config=ipu_run_config.RunConfig())

    with self.assertRaisesRegexp(ValueError, "must return Dataset"):
      estimator.train(input_fn=my_input_fn, steps=1)

  def testVerifyTrainFeeds(self):
    def my_model_fn(features, labels, mode):
      self.assertEquals(model_fn_lib.ModeKeys.TRAIN, mode)

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

    self.assertEquals(session_run_counter.num_session_runs, num_steps)

    model_dir = estimator.model_dir
    events_file = glob.glob(model_dir + "/*tfevents*")
    assert len(events_file) == 1
    events_file = events_file[0]
    losses = list()
    for e in summary_iterator.summary_iterator(events_file):
      for v in e.summary.value:
        if "loss" in v.tag:
          losses.append(v.simple_value)

    self.assertEqual(losses, list(np.arange(0.0, 12.0, 2.0)))

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
      self.assertEquals(model_fn_lib.ModeKeys.TRAIN, mode)

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

    self.assertEquals(six.viewkeys(estimator_variables),
                      six.viewkeys(warm_started_estimator_variables))

    for k, _ in six.iteritems(estimator_variables):
      self.assertEquals(estimator_variables[k][0],
                        warm_started_estimator_variables[k][0])

  def testCompileSummary(self):
    def my_model_fn(features, labels, mode):
      self.assertEquals(model_fn_lib.ModeKeys.TRAIN, mode)

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

    ipu_options = ipu_utils.create_ipu_config(profiling=True)

    ipu_config = ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                             ipu_options=ipu_options,
                                             compile_summary=True)

    run_config = ipu_run_config.RunConfig(ipu_run_config=ipu_config,
                                          log_step_count_steps=1)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                           config=run_config)

    estimator.train(input_fn=my_input_fn, steps=4)

    event_file = glob.glob(estimator.model_dir + "/event*")
    self.assertTrue(len(event_file) == 1)
    compile_for_ipu_count = 0
    for summary in summary_iterator.summary_iterator(event_file[0]):
      for val in summary.summary.value:
        if val.tag == "ipu_trace":
          for evt_str in val.tensor.string_val:
            evt = IpuTraceEvent.FromString(evt_str)

            if (evt.type == IpuTraceEvent.COMPILE_END
                and len(evt.compile_end.compilation_report)) > 0:
              compile_for_ipu_count += 1

    self.assertEqual(compile_for_ipu_count, 1)

  def testEventDecode(self):
    class EventTraceHook(session_run_hook.SessionRunHook):
      def __init__(self):
        self._event_op = None
        self._events = None

      def begin(self):
        self._event_op = gen_ipu_ops.ipu_event_trace()

      def after_run(self, run_context, run_values):
        self._events = run_context.session.run(self._event_op)

      @property
      def events(self):
        return self._events

    def my_model_fn(features, labels, mode):
      self.assertEquals(model_fn_lib.ModeKeys.TRAIN, mode)

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

    ipu_options = ipu_utils.create_ipu_config(profiling=True)

    ipu_config = ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                             ipu_options=ipu_options,
                                             compile_summary=True)

    run_config = ipu_run_config.RunConfig(ipu_run_config=ipu_config,
                                          log_step_count_steps=1)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn,
                                           config=run_config)

    event_trace_hook = EventTraceHook()
    hooks = [event_trace_hook]
    estimator.train(input_fn=my_input_fn, steps=4, hooks=hooks)

    events = event_trace_hook.events
    self.assertEqual(len(events), 3)


if __name__ == "__main__":
  googletest.main()
