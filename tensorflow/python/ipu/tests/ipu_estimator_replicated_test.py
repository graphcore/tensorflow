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
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.keras import layers
from tensorflow.python import ipu
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import ipu_estimator
from tensorflow.python.ipu import ipu_run_config
from tensorflow.python.ipu import ipu_session_run_hooks
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu.ops import replication_ops
from tensorflow.python.ipu.optimizers import cross_replica_optimizer
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer
from tensorflow.python.ipu.optimizers import sharded_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import googletest
from tensorflow.python.summary import summary_iterator
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import session_run_hook


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


class IPUEstimatorReplicatedTest(test_util.TensorFlowTestCase,
                                 parameterized.TestCase):
  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testTrainReplicated(self):
    def my_model_fn(features, labels, mode):  # pylint: disable=unused-argument
      self.assertEqual(model_fn_lib.ModeKeys.TRAIN, mode)

      loss = ipu.ops.cross_replica_ops.cross_replica_sum(features, name="loss")

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

    ipu_options = ipu.config.IPUConfig()
    ipu_options.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(ipu_options)
    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                                   num_replicas=4,
                                                   ipu_options=ipu_options),
        log_step_count_steps=1,
        save_summary_steps=1)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)

    session_run_counter = _SessionRunCounter()

    num_steps = 6
    estimator.train(input_fn=my_input_fn,
                    steps=num_steps,
                    hooks=[session_run_counter])

    self.assertEqual(session_run_counter.num_session_runs,
                     num_steps // config.ipu_run_config.iterations_per_loop)

    model_dir = estimator.model_dir
    events_file = glob.glob(model_dir + "/*tfevents*")
    assert len(events_file) == 1
    events_file = events_file[0]
    loss_output = list()
    for e in summary_iterator.summary_iterator(events_file):
      for v in e.summary.value:
        if "loss" in v.tag:
          loss_output.append(v.simple_value)

    # loss is averaged across iterations per loop
    self.assertAllClose(loss_output, [14.0, 16.0, 18.0])

  @parameterized.parameters(
      {'use_gradient_accumulation': False},
      {'use_gradient_accumulation': True},
  )
  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testTrainReplicatedOnRegressionDataset(self, use_gradient_accumulation):
    iterations_per_loop = 2

    def my_model_fn(features, labels, mode):
      self.assertEqual(model_fn_lib.ModeKeys.TRAIN, mode)

      with variable_scope.variable_scope("vs", use_resource=True):
        predictions = layers.Dense(units=1)(features)

      loss = losses.mean_squared_error(labels=labels, predictions=predictions)
      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      sharded_optimizer_obj = sharded_optimizer.ShardedOptimizer(optimizer)

      if use_gradient_accumulation:
        cross_replica_optimizer_obj = gradient_accumulation_optimizer.\
            CrossReplicaGradientAccumulationOptimizerV2(
                sharded_optimizer_obj, iterations_per_loop)
      else:
        cross_replica_optimizer_obj = \
          cross_replica_optimizer.CrossReplicaOptimizer(sharded_optimizer_obj)

      train_op = cross_replica_optimizer_obj.minimize(loss)

      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op)

    def my_input_fn():
      dataset = dataset_ops.Dataset.from_tensor_slices(
          _create_regression_dataset(num_samples=1000, num_features=5))
      dataset = dataset.batch(batch_size=2, drop_remainder=True).repeat()
      return dataset

    ipu_options = ipu.config.IPUConfig()
    ipu_options.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(ipu_options)
    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(
            iterations_per_loop=iterations_per_loop,
            num_replicas=4,
            ipu_options=ipu_options),
        log_step_count_steps=1,
        save_summary_steps=1)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)

    session_run_counter = _SessionRunCounter()

    num_steps = 6
    estimator.train(input_fn=my_input_fn,
                    steps=num_steps,
                    hooks=[session_run_counter])

    self.assertEqual(session_run_counter.num_session_runs,
                     num_steps // config.ipu_run_config.iterations_per_loop)

    model_dir = estimator.model_dir
    events_file = glob.glob(model_dir + "/*tfevents*")
    assert len(events_file) == 1
    events_file = events_file[0]
    loss_output = list()
    for e in summary_iterator.summary_iterator(events_file):
      for v in e.summary.value:
        if "loss" in v.tag:
          loss_output.append(v.simple_value)

    self.assertTrue(loss_output[0] > loss_output[-1])

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testShardedAndReplicatedTraining(self):
    def my_model_fn(features, labels, mode):
      self.assertEqual(model_fn_lib.ModeKeys.TRAIN, mode)

      with variable_scope.variable_scope("vs", use_resource=True):
        with ipu.scopes.ipu_shard(0):
          out_0 = layers.Dense(units=1)(features)

        with ipu.scopes.ipu_shard(1):
          predictions = layers.Dense(units=1)(out_0)

      loss = losses.mean_squared_error(labels=labels, predictions=predictions)
      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      sharded_optimizer_obj = sharded_optimizer.ShardedOptimizer(optimizer)
      cross_replica_optimizer_obj = \
      cross_replica_optimizer.CrossReplicaOptimizer(
          sharded_optimizer_obj)
      train_op = cross_replica_optimizer_obj.minimize(loss)

      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op)

    def my_input_fn():
      dataset = dataset_ops.Dataset.from_tensor_slices(
          _create_regression_dataset(num_samples=1000, num_features=5))
      dataset = dataset.batch(batch_size=2, drop_remainder=True).repeat()
      return dataset

    ipu_options = ipu.config.IPUConfig()
    ipu_options.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(ipu_options)
    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(iterations_per_loop=2,
                                                   num_replicas=2,
                                                   num_shards=2,
                                                   ipu_options=ipu_options),
        log_step_count_steps=1,
        save_summary_steps=1)

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)

    session_run_counter = _SessionRunCounter()

    num_steps = 10
    estimator.train(input_fn=my_input_fn,
                    steps=num_steps,
                    hooks=[session_run_counter])

    self.assertEqual(session_run_counter.num_session_runs,
                     num_steps // config.ipu_run_config.iterations_per_loop)

    model_dir = estimator.model_dir
    events_file = glob.glob(model_dir + "/*tfevents*")
    assert len(events_file) == 1
    events_file = events_file[0]
    loss_output = list()
    for e in summary_iterator.summary_iterator(events_file):
      for v in e.summary.value:
        if "loss" in v.tag:
          loss_output.append(v.simple_value)

    self.assertTrue(loss_output[0] > loss_output[-1])

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testReplicatedEvaluation(self):
    def my_input_fn():
      # IPU0 mean: 2, max: 3
      # IPU1 mean: 4, max: 5
      features = [
          [1.0],  # IPU0
          [3.0],  # IPU0
          [5.0],  # IPU1
          [3.0],  # IPU1
          [1.0],  # IPU2
          [3.0],  # IPU2
          [5.0],  # IPU3
          [3.0],  # IPU3
      ]
      return dataset_ops.Dataset.from_tensor_slices(features).batch(
          batch_size=2, drop_remainder=True)

    def my_model_fn(features, mode):
      loss = math_ops.reduce_max(features)
      eval_metric_ops = {
          "feature_mean": metrics_impl.mean(features),
      }
      return model_fn_lib.EstimatorSpec(mode,
                                        loss=loss,
                                        eval_metric_ops=eval_metric_ops)

    ipu_options = ipu.config.IPUConfig()
    ipu_options.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(ipu_options)
    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(
            iterations_per_loop=1, num_replicas=4, ipu_options=ipu_options))

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)
    scores = estimator.evaluate(my_input_fn, steps=1)
    self.assertEqual(3., scores["feature_mean"])
    self.assertEqual(4., scores[model_fn_lib.LOSS_METRIC_KEY])

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testReplicatedEvaluationOnHost(self):
    def my_input_fn():
      features = [0, 0, 0, 1, 0, 0, 0, 1]
      labels = [0, 1, 0, 1, 0, 1, 0, 1]
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
          "mse": metrics_impl.mean_squared_error(labels, features),
          "rmse": metrics_impl.root_mean_squared_error(labels, features),
      }

    def my_model_fn(features, labels, mode):
      loss = math_ops.cast(replication_ops.replication_index(), np.float32)
      eval_metrics = (my_metrics_fn, [features, labels])
      return ipu_estimator.IPUEstimatorSpec(mode,
                                            loss=loss,
                                            eval_metrics=eval_metrics)

    ipu_options = ipu.config.IPUConfig()
    ipu_options.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(ipu_options)
    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(
            iterations_per_loop=1, num_replicas=4, ipu_options=ipu_options))

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)
    scores = estimator.evaluate(my_input_fn, steps=1)
    self.assertEqual(0.75, scores["accuracy"])
    self.assertEqual(1.0, scores["precision"])
    self.assertEqual(0.5, scores["recall"])
    self.assertEqual(0.5, scores["recall_at_1"])
    self.assertEqual(1.0, scores["recall_at_2"])
    self.assertEqual(0.25, scores["mse"])
    self.assertEqual(0.5, scores["rmse"])
    self.assertEqual(1.5, scores[model_fn_lib.LOSS_METRIC_KEY])

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testReplicatedPrediction(self):
    def my_input_fn():
      features = [
          [1.0],  # IPU0
          [3.0],  # IPU0
          [5.0],  # IPU1
          [3.0],  # IPU1
          [7.0],  # IPU2
          [3.0],  # IPU2
          [9.0],  # IPU3
          [3.0],  # IPU3
      ]
      return dataset_ops.Dataset.from_tensor_slices(features).batch(
          batch_size=2, drop_remainder=True)

    hook = ipu_session_run_hooks.IPULoggingTensorHook(every_n_iter=1)

    def my_model_fn(features, mode):
      logging_op = hook.log({"features": features})
      with ops.control_dependencies([logging_op]):
        predictions = math_ops.reduce_max(features)

      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=predictions,
      )

    ipu_options = ipu.config.IPUConfig()
    ipu_options.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(ipu_options)
    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(
            iterations_per_loop=1, num_replicas=4, ipu_options=ipu_options))
    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)

    outputs = estimator.predict(input_fn=my_input_fn,
                                yield_single_examples=True)
    self.assertEqual(3.0, next(outputs))
    self.assertEqual(5.0, next(outputs))

    outputs = estimator.predict(input_fn=my_input_fn,
                                yield_single_examples=False,
                                hooks=[hook])
    self.assertAllEqual([3.0, 5.0, 7.0, 9.0], next(outputs))

  @tu.test_uses_ipus(num_ipus=4)
  @test_util.deprecated_graph_mode_only
  def testReplicatedTrainingWithoutCrossReplicaSumShouldThrow(self):
    def my_input_fn():
      return dataset_ops.Dataset.from_tensor_slices([])

    def my_model_fn(features, mode):
      loss = math_ops.reduce_sum(features)
      train_op = array_ops.identity(loss)
      return model_fn_lib.EstimatorSpec(mode, loss=loss, train_op=train_op)

    ipu_options = ipu.config.IPUConfig()
    ipu_options.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(ipu_options)
    config = ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config.IPURunConfig(
            iterations_per_loop=1, num_replicas=4, ipu_options=ipu_options))
    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)

    with self.assertRaisesRegex(
        ValueError, "This is not a valid replicated training graph"):
      estimator.train(input_fn=my_input_fn, steps=1)


if __name__ == "__main__":
  googletest.main()
