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

import collections
import glob
import json
import multiprocessing
import os

import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.compat.v1 import disable_v2_behavior
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import keras
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.cluster_resolver.tfconfig_cluster_resolver import TFConfigClusterResolver
from tensorflow.python.distribute.reduce_util import ReduceOp
from tensorflow.python.estimator import estimator_lib
from tensorflow.python.framework import ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_estimator
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import ipu_pipeline_estimator
from tensorflow.python.ipu import ipu_run_config
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import scopes
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.ipu_multi_worker_strategy import IPUMirroredVariable
from tensorflow.python.ipu.ipu_multi_worker_strategy import IPUMultiWorkerStrategyV1
from tensorflow.python.ipu.ipu_multi_worker_strategy import IPUSyncOnReadVariable
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.summary import summary_iterator
from tensorflow.python.training import server_lib
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.python.training.momentum import MomentumOptimizer
from tensorflow.python.training.monitored_session import MonitoredTrainingSession

disable_v2_behavior()


class IPUMultiWorkerStrategyV1Test(multi_worker_test_base.MultiWorkerTestBase):
  """Tests using multiple threads in the same processes."""
  @classmethod
  def setUpClass(cls):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    cls._num_workers = 2
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=cls._num_workers, num_ps=0, has_chief=False)

  def setUp(self):
    # We use a different key_base for each test so that collective keys won't be
    # reused.
    IPUMultiWorkerStrategyV1._collective_key_base += 100000
    super().setUp()

  def _create_test_objects(self, task_type, task_id, variables_on_host=True):
    sess_config = config_pb2.ConfigProto()
    sess_config.allow_soft_placement = False
    sess_config.log_device_placement = False

    cluster_spec = multi_worker_util.normalize_cluster_spec(self._cluster_spec)
    cluster_resolver = SimpleClusterResolver(cluster_spec=cluster_spec,
                                             task_type=task_type,
                                             task_id=task_id)
    target = cluster_resolver.master(task_id=task_id,
                                     task_type=task_type,
                                     rpc_layer="grpc")
    strategy = IPUMultiWorkerStrategyV1(cluster_resolver,
                                        variables_on_host=variables_on_host)
    sess_config = strategy.update_config_proto(sess_config)

    return strategy, target, sess_config

  def _get_devices(self, task_type, task_id):
    cpu_device = "/job:{}/replica:0/task:{}/device:CPU:0".format(
        task_type, task_id)

    ipu_device = "/job:{}/replica:0/task:{}/device:IPU:0".format(
        task_type, task_id)

    return cpu_device, ipu_device

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_strategy_first_worker(self):
    strategy, _, _ = self._create_test_objects(task_type="worker", task_id=0)
    self.assertEqual(2, strategy.num_replicas_in_sync)
    self.assertEqual(True, strategy.extended.experimental_between_graph)
    self.assertEqual(True, strategy.extended.experimental_should_init)
    self.assertEqual(True, strategy.extended.should_checkpoint)
    self.assertEqual(True, strategy.extended.should_save_summary)

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_strategy_second_worker(self):
    strategy, _, _ = self._create_test_objects(task_type="worker", task_id=1)
    self.assertEqual(2, strategy.num_replicas_in_sync)
    self.assertEqual(True, strategy.extended.experimental_between_graph)
    self.assertEqual(True, strategy.extended.experimental_should_init)
    self.assertEqual(False, strategy.extended.should_checkpoint)
    self.assertEqual(False, strategy.extended.should_save_summary)

  def test_initializer_colocation(self):
    strategy, _, _ = self._create_test_objects(task_type="worker", task_id=0)

    with strategy.scope():
      v = variables.Variable(1.0)

    assign_op = v.initializer.control_inputs[0]

    # The first input is the variable, the second is the value.
    initial_value_op = assign_op.inputs[1].op

    # The initial value should be colocated with the CPU.
    self.assertEqual(initial_value_op.colocation_groups(), [b'loc:@cpu'])

  def _test_variables_on_host(self, task_type, task_id, _num_gpus):
    strategy, _, _ = self._create_test_objects(task_type,
                                               task_id,
                                               variables_on_host=True)
    cpu_device, ipu_device = self._get_devices(task_type, task_id)

    with strategy.scope():

      v = variables.Variable(1.0)
      self.assertEqual(cpu_device, v.device)

      def per_replica_fn():
        w = variable_scope.get_variable(name="w", initializer=0.0)
        self.assertEqual(cpu_device, w.device)
        op = math_ops.abs(w)
        self.assertEqual(ipu_device, op.device)
        return op

      per_replica_op = strategy.run(per_replica_fn)
      self.assertEqual(ipu_device, per_replica_op.device)

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_variables_on_host(self):
    self._run_between_graph_clients(self._test_variables_on_host,
                                    self._cluster_spec,
                                    num_gpus=0)

  def _test_variables_on_ipu(self, task_type, task_id, _num_gpus):
    strategy, _, _ = self._create_test_objects(task_type,
                                               task_id,
                                               variables_on_host=False)
    _, ipu_device = self._get_devices(task_type, task_id)

    with strategy.scope():

      v = variables.Variable(1.0)
      self.assertEqual(ipu_device, v.device)

      def per_replica_fn():
        w = variable_scope.get_variable(name="w", initializer=0.0)
        self.assertEqual(ipu_device, w.device)
        op = math_ops.abs(w)
        self.assertEqual(ipu_device, op.device)
        return op

      per_replica_op = strategy.run(per_replica_fn)
      self.assertEqual(ipu_device, per_replica_op.device)

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_variables_on_ipu(self):
    self._run_between_graph_clients(self._test_variables_on_ipu,
                                    self._cluster_spec,
                                    num_gpus=0)

  def _test_all_reduce(self, task_type, task_id, _num_gpus):
    strategy, target, sess_config = self._create_test_objects(
        task_type=task_type, task_id=task_id)

    variable_device, compute_device = self._get_devices(task_type, task_id)

    with strategy.scope():

      def per_replica_fn(x):
        with ops.device("/device:IPU:0"):
          y = x * x
          self.assertEqual(compute_device, y.device)
          return y

      inputs = array_ops.placeholder(dtype=np.float32, shape=())
      per_replica_y = strategy.run(per_replica_fn, args=[inputs])
      self.assertEqual(compute_device, per_replica_y.device)
      sum_y = strategy.reduce(ReduceOp.SUM, per_replica_y, axis=None)
      self.assertEqual(variable_device, sum_y.device)

      with session_lib.Session(target=target, config=sess_config) as sess:
        out = sess.run(sum_y, feed_dict={inputs: task_id + 1})
        self.assertEqual(5.0, out)  # 1*1 + 2*2

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_all_reduce(self):
    self._run_between_graph_clients(self._test_all_reduce,
                                    self._cluster_spec,
                                    num_gpus=0)

  def _test_mirrored_variable(self, task_type, task_id, _num_gpus):
    strategy, target, sess_config = self._create_test_objects(
        task_type=task_type, task_id=task_id)

    variable_device, compute_device = self._get_devices(task_type, task_id)

    with strategy.scope():

      def per_replica_fn():
        with ops.device("/device:IPU:0"):
          w0 = variable_scope.get_variable(name="w0", initializer=task_id + 1)
          self.assertIsInstance(w0, IPUMirroredVariable)
          self.assertEqual(variable_device, w0.device)
          cached_value = w0.value()
          self.assertEqual(compute_device, cached_value.device)
          ret = w0 * w0
          self.assertEqual(compute_device, ret.device)
          return ret

      per_replica_ret = strategy.run(per_replica_fn, args=[])
      self.assertEqual(compute_device, per_replica_ret.device)
      sum_ret = strategy.reduce(ReduceOp.SUM, per_replica_ret, axis=None)
      self.assertEqual(variable_device, sum_ret.device)

      with session_lib.Session(target=target, config=sess_config) as sess:
        sess.run(variables.global_variables_initializer())
        # Both should have initial value from first worker
        self.assertEqual([1.0], sess.run(variables.global_variables()))
        self.assertEqual(2.0, sess.run(sum_ret))  # 1*1 + 1*1

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_mirrored_variable(self):
    self._run_between_graph_clients(self._test_mirrored_variable,
                                    self._cluster_spec,
                                    num_gpus=0)

  def _test_sync_on_read_variable(self, task_type, task_id, _num_gpus):
    strategy, target, sess_config = self._create_test_objects(
        task_type=task_type, task_id=task_id)

    variable_device, compute_device = self._get_devices(task_type, task_id)

    with strategy.scope():

      def per_replica_fn(x):
        with ops.device("/device:IPU:0"):
          w0 = variable_scope.get_variable(
              name="w0",
              initializer=float(task_id + 1),
              synchronization=variable_scope.VariableSynchronization.ON_READ,
              aggregation=variable_scope.VariableAggregation.MEAN)
          self.assertIsInstance(w0, IPUSyncOnReadVariable)
          self.assertEqual(compute_device, w0.device)
          initializer_tensor = w0.values[0].initializer.inputs[1]
          self.assertEqual(variable_device, initializer_tensor.device)
          return w0.assign_add(x)

      inputs = array_ops.placeholder(dtype=np.float32, shape=())
      assign_add_op = strategy.run(per_replica_fn, args=[inputs])

      with session_lib.Session(target=target, config=sess_config) as sess:
        sess.run(variables.global_variables_initializer())
        # Both should have initial value from first worker
        self.assertEqual([1.0], sess.run(variables.global_variables()))
        sess.run(assign_add_op, feed_dict={inputs: task_id + 1})
        # mean(1 + 1, 1 + 2) = 2.5
        self.assertEqual([2.5], sess.run(variables.global_variables()))

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_sync_on_read_variable(self):
    self._run_between_graph_clients(self._test_sync_on_read_variable,
                                    self._cluster_spec,
                                    num_gpus=0)

  def _test_train_split_device_host_fn(self, task_type, task_id, _num_gpus):
    strategy, target, sess_config = self._create_test_objects(
        task_type=task_type, task_id=task_id)

    variable_device, compute_device = self._get_devices(task_type, task_id)

    with strategy.scope():
      learning_rate = 0.5
      initial_w = 2.0
      optimizer = GradientDescentOptimizer(learning_rate)

      def device_step_fn(x):
        w = variable_scope.get_variable(name="w", initializer=initial_w)
        self.assertEqual(variable_device, w.device)
        self.assertEqual(compute_device, w.value().device)

        loss = w * x
        self.assertEqual(compute_device, loss.device)

        with ops.name_scope("compute_gradients"):
          grads_and_vars = optimizer.compute_gradients(loss)
          grads = [g for (g, _) in grads_and_vars]

        return grads, loss

      def compiled_device_step_fn(inputs):
        with ipu_scope("/device:IPU:0"):
          grads, loss = ipu_compiler.compile(device_step_fn, inputs=[inputs])
          return grads, loss

      def host_step_fn(grads):
        with ops.name_scope("apply_gradients"):
          grads_and_vars = zip(grads, variables.global_variables())
          train_op = optimizer.apply_gradients(grads_and_vars)
          self.assertEqual(variable_device, train_op.device)
          return train_op

      def step_fn(inputs):
        grads, loss = compiled_device_step_fn(inputs)
        with ops.device("/device:CPU:0"):
          train_op = host_step_fn(grads)
        return train_op, loss

      inputs = array_ops.placeholder(dtype=np.float32, shape=())
      train_op, per_replica_loss = strategy.run(step_fn, args=[inputs])
      self.assertEqual(compute_device, per_replica_loss.device)
      total_loss = strategy.reduce(ReduceOp.SUM, per_replica_loss, axis=None)
      self.assertEqual(variable_device, total_loss.device)

      with session_lib.Session(target=target, config=sess_config) as sess:
        sess.run(variables.global_variables_initializer())

        # L(x) = num_replicas * w * x
        # dL(x)/dw = num_replicas * x
        # w := w - learning_rate * num_replicas * x

        num_replicas = strategy.num_replicas_in_sync
        reference_w = initial_w
        w_tensor = variables.global_variables()[0]
        for x in range(10):
          self.assertEqual(reference_w, sess.run(w_tensor))
          _, loss_val = sess.run([train_op, total_loss], feed_dict={inputs: x})
          self.assertEqual(num_replicas * reference_w * x, loss_val)
          reference_w -= learning_rate * num_replicas * x

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_train_split_device_host_fn(self):
    self._run_between_graph_clients(self._test_train_split_device_host_fn,
                                    self._cluster_spec,
                                    num_gpus=0)

  def _test_train_combined_device_host_fn(self, task_type, task_id, _num_gpus):
    strategy, target, sess_config = self._create_test_objects(
        task_type=task_type, task_id=task_id)

    variable_device, compute_device = self._get_devices(task_type, task_id)

    with strategy.scope():
      learning_rate = 0.5
      initial_w = 2.0
      optimizer = GradientDescentOptimizer(learning_rate)

      def step_fn(x):
        with ipu_scope("/device:IPU:0"):
          w = variable_scope.get_variable(name="w", initializer=initial_w)
          self.assertEqual(variable_device, w.device)
          self.assertEqual(compute_device, w.value().device)

          loss = w * x
          self.assertEqual(compute_device, loss.device)
          # optimizer.apply_gradients() is colocated with the variables even
          # in ipu_scope, while optimizer.compute_gradients() is not.
          train_op = optimizer.minimize(loss)
          self.assertEqual(variable_device, train_op.device)
          return train_op, loss

      inputs = array_ops.placeholder(dtype=np.float32, shape=())
      train_op, per_replica_loss = strategy.run(step_fn, args=[inputs])
      self.assertEqual(compute_device, per_replica_loss.device)
      total_loss = strategy.reduce(ReduceOp.SUM, per_replica_loss, axis=None)
      self.assertEqual(variable_device, total_loss.device)

      with session_lib.Session(target=target, config=sess_config) as sess:
        sess.run(variables.global_variables_initializer())

        # L(x) = num_replicas * w * x
        # dL(x)/dw = num_replicas * x
        # w := w - learning_rate * num_replicas * x

        num_replicas = strategy.num_replicas_in_sync
        reference_w = initial_w
        w_tensor = variables.global_variables()[0]
        for x in range(10):
          self.assertEqual(reference_w, sess.run(w_tensor))
          _, loss_val = sess.run([train_op, total_loss], feed_dict={inputs: x})
          self.assertEqual(num_replicas * reference_w * x, loss_val)
          reference_w -= learning_rate * num_replicas * x

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_train_combined_device_host_fn(self):
    self._run_between_graph_clients(self._test_train_combined_device_host_fn,
                                    self._cluster_spec,
                                    num_gpus=0)

  def _test_slot_variable_on_host(self, task_type, task_id, _num_gpus):
    strategy, target, sess_config = self._create_test_objects(
        task_type=task_type, task_id=task_id)

    variable_device, _ = self._get_devices(task_type, task_id)

    with strategy.scope():
      optimizer = MomentumOptimizer(learning_rate=0.5, momentum=0.9)

      def step_fn(x):
        with ipu_scope("/device:IPU:0"):
          w = variable_scope.get_variable(name="w", initializer=1.0)
          loss = w * x
          train_op = optimizer.minimize(loss)
          return train_op, loss

      inputs = array_ops.placeholder(dtype=np.float32, shape=())
      train_op, per_replica_loss = strategy.run(step_fn, args=[inputs])
      total_loss = strategy.reduce(ReduceOp.SUM, per_replica_loss, axis=None)

      # Verify device placement of momentum accumulator variable.
      self.assertEqual(1, len(optimizer.variables()))
      self.assertEqual(variable_device, optimizer.variables()[0].device)

      with session_lib.Session(target=target, config=sess_config) as sess:
        sess.run(variables.global_variables_initializer())
        _, loss_val = sess.run([train_op, total_loss], feed_dict={inputs: 1.0})
        self.assertEqual(2.0, loss_val)

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_slot_variable_on_host(self):
    self._run_between_graph_clients(self._test_slot_variable_on_host,
                                    self._cluster_spec,
                                    num_gpus=0)

  def _test_slot_variable_on_ipu(self, task_type, task_id, _num_gpus):
    strategy, target, sess_config = self._create_test_objects(
        task_type=task_type, task_id=task_id, variables_on_host=False)

    _, ipu_device = self._get_devices(task_type, task_id)

    with strategy.scope():
      optimizer = MomentumOptimizer(learning_rate=0.5, momentum=0.9)

      def step_fn(x):
        with ipu_scope("/device:IPU:0"):
          w = variable_scope.get_variable(name="w", initializer=1.0)
          loss = w * x
          train_op = optimizer.minimize(loss)
          return train_op, loss

      inputs = array_ops.placeholder(dtype=np.float32, shape=())
      train_op, per_replica_loss = strategy.run(step_fn, args=[inputs])
      total_loss = strategy.reduce(ReduceOp.SUM, per_replica_loss, axis=None)

      # Verify device placement of momentum accumulator variable.
      self.assertEqual(1, len(optimizer.variables()))
      self.assertEqual(ipu_device, optimizer.variables()[0].device)

      with session_lib.Session(target=target, config=sess_config) as sess:
        sess.run(variables.global_variables_initializer())
        _, loss_val = sess.run([train_op, total_loss], feed_dict={inputs: 1.0})
        self.assertEqual(2.0, loss_val)

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_slot_variable_on_ipu(self):
    self._run_between_graph_clients(self._test_slot_variable_on_ipu,
                                    self._cluster_spec,
                                    num_gpus=0)

  def _test_distribute_dataset(self, task_type, task_id, _num_gpus):
    strategy, target, sess_config = self._create_test_objects(
        task_type=task_type, task_id=task_id)

    with strategy.scope():

      def step_fn(x):
        with ipu_scope("/device:IPU:0"):
          y = x.values[0] * x.values[0]
          return y

      dataset = dataset_ops.Dataset.range(10)
      dataset = dataset.map(lambda x: math_ops.cast(x, np.float32))
      dataset = dataset.batch(2, drop_remainder=True)  # global batch size
      dist_dataset = strategy.experimental_distribute_dataset(dataset)
      inputs = dist_dataset.make_initializable_iterator()
      per_replica_y = strategy.run(step_fn, args=[next(inputs)])
      sum_y = strategy.reduce(ReduceOp.SUM, per_replica_y, axis=None)

      with session_lib.Session(target=target, config=sess_config) as sess:
        sess.run(inputs.initializer)
        self.assertEqual(1.0, sess.run(sum_y))  # 0*0 + 1*1
        self.assertEqual(13.0, sess.run(sum_y))  # 2*2 + 3*3
        self.assertEqual(41.0, sess.run(sum_y))  # 4*4 + 5*5

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_distribute_dataset(self):
    self._run_between_graph_clients(self._test_distribute_dataset,
                                    self._cluster_spec,
                                    num_gpus=0)

  def _test_monitored_training_session(self, task_type, task_id, _num_gpus):
    strategy, target, sess_config = self._create_test_objects(
        task_type=task_type, task_id=task_id)

    with strategy.scope():

      def step_fn(x):
        with ipu_scope("/device:IPU:0"):
          w = variable_scope.get_variable("w", initializer=2.0)
          y = w * x
          return y

      inputs = array_ops.placeholder(dtype=np.float32, shape=())
      per_replica_y = strategy.run(step_fn, args=[inputs])
      sum_y = strategy.reduce(ReduceOp.SUM, per_replica_y, axis=None)

      with MonitoredTrainingSession(master=target, config=sess_config) as sess:
        out = sess.run(sum_y, feed_dict={inputs: task_id + 1})
        self.assertEqual(6.0, out)  # 2*1 + 2*2

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_monitored_training_session(self):
    self._run_between_graph_clients(self._test_monitored_training_session,
                                    self._cluster_spec,
                                    num_gpus=0)

  def _test_ipu_estimator_train_with_host_call(self, task_type, task_id,
                                               _num_gpus):
    strategy, target, _ = self._create_test_objects(task_type=task_type,
                                                    task_id=task_id)

    learning_rate = 0.5
    initial_w = 2.0
    # Use momentum, but set to zero, just to verify that the
    # momentum accumulator "slot" does not cause any problems.
    optimizer = MomentumOptimizer(learning_rate=learning_rate, momentum=0.0)

    def host_model_fn(*grads):
      grads_and_vars = zip(grads, variables.trainable_variables())
      with ops.name_scope("apply_gradients"):
        train_op = optimizer.apply_gradients(grads_and_vars)
        return train_op

    def my_model_fn(features, labels, mode):
      w = variable_scope.get_variable(name="w", initializer=initial_w)
      predictions = features * w
      loss = losses.mean_squared_error(labels=labels, predictions=predictions)

      # Note: According to some comments, this might be subject to change in TF2.
      # Remember to update documentation and examples when this happens.
      self.assertEqual(ReduceOp.MEAN, distribute_lib.get_loss_reduction())

      with ops.name_scope("compute_gradients"):
        grads_and_vars = optimizer.compute_gradients(loss)
      grads = [g for (g, _) in grads_and_vars]
      train_op = array_ops.identity(loss)
      host_call = (host_model_fn, grads)

      return ipu_estimator.IPUEstimatorSpec(mode=mode,
                                            loss=loss,
                                            train_op=train_op,
                                            host_call=host_call)

    features = np.array([[1.0], [2.0]], dtype=np.float32)
    labels = np.array([[1.0], [2.0]], dtype=np.float32)

    def my_input_fn(input_context):
      dataset = dataset_ops.Dataset.from_tensor_slices((features, labels))
      dataset = dataset.batch(1, drop_remainder=True).repeat()

      num_shards = input_context.num_input_pipelines
      shard_index = input_context.input_pipeline_id
      self.assertEqual(self._num_workers, num_shards)
      self.assertEqual(task_id, shard_index)

      dataset = dataset.shard(num_shards=num_shards, index=shard_index)
      return dataset

    config = ipu_run_config.RunConfig(
        session_config=config_pb2.ConfigProto(allow_soft_placement=False),
        master=target,
        train_distribute=strategy,
    )

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)

    reference_w = initial_w

    for i in range(3):
      estimator.train(my_input_fn, steps=1)
      self.assertEqual(i + 1, estimator.get_variable_value("global_step"))

      # L(x, y) = 0.5 * ((w * x_0 - y_0)^2 + (w * x_1 - y_1)^2)
      # dL(x, y)/dw = (w * x_0 - y_0) * x_0 + (w * x_1 - y_1) * x_1
      reference_gradient = np.sum((reference_w * features - labels) * features)
      reference_w -= learning_rate * reference_gradient
      self.assertEqual(reference_w, estimator.get_variable_value("w"))

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_ipu_estimator_train_with_host_call(self):
    self._run_between_graph_clients(
        self._test_ipu_estimator_train_with_host_call,
        self._cluster_spec,
        num_gpus=0)

  def _test_batch_normalization(self, task_type, task_id, _num_gpus):
    strategy, target, sess_config = self._create_test_objects(
        task_type=task_type, task_id=task_id, variables_on_host=True)

    variable_device, compute_device = self._get_devices(task_type, task_id)

    with strategy.scope():
      batch_norm = BatchNormalization(momentum=0.0)

      def per_replica_fn(x):
        with ops.device("/device:IPU:0"):
          y = batch_norm(x, training=True)

          self.assertIsInstance(batch_norm.beta, IPUMirroredVariable)
          self.assertIsInstance(batch_norm.gamma, IPUMirroredVariable)
          self.assertEqual(variable_device, batch_norm.beta.device)
          self.assertEqual(variable_device, batch_norm.gamma.device)

          self.assertIsInstance(batch_norm.moving_mean, IPUSyncOnReadVariable)
          self.assertIsInstance(batch_norm.moving_variance,
                                IPUSyncOnReadVariable)
          self.assertEqual(compute_device, batch_norm.moving_mean.device)
          self.assertEqual(compute_device, batch_norm.moving_variance.device)

          return y

      def compiled_per_replica_fn(inputs):
        with ipu_scope("/device:IPU:0"):
          [out] = ipu_compiler.compile(per_replica_fn, inputs=[inputs])
          return out

      inputs = array_ops.placeholder(dtype=np.float32, shape=(2, 1))
      per_replica_y = strategy.run(compiled_per_replica_fn, args=[inputs])
      sum_y = strategy.reduce(ReduceOp.SUM, per_replica_y, axis=None)
      self.assertEqual(variable_device, batch_norm.moving_mean._get().device)  # pylint: disable=protected-access

      with session_lib.Session(target=target, config=sess_config) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(sum_y, feed_dict={inputs: [[2.0 * (task_id + 1)], [0.0]]})
        task_local_mean = batch_norm.moving_mean._get_on_device_or_primary()  # pylint: disable=protected-access
        self.assertAllEqual([task_id + 1], sess.run(task_local_mean))

        # mean(mean(2, 0), mean(4, 0)) = mean(1, 3) = 1.5
        global_mean = batch_norm.moving_mean
        self.assertAllEqual([1.5], sess.run(global_mean))

        # mean(var(2, 0), var(4, 0)) = mean(1, 4) = 2.5
        self.assertAllEqual([2.5], sess.run(batch_norm.moving_variance))

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_batch_normalization(self):
    self._run_between_graph_clients(self._test_batch_normalization,
                                    self._cluster_spec,
                                    num_gpus=0)


def _get_executed_nodes_by_device(run_metadata):
  nodes_by_device = collections.defaultdict(list)
  for dev_stats in run_metadata.step_stats.dev_stats:
    for node_stats in dev_stats.node_stats:
      nodes_by_device[dev_stats.device].append(node_stats.node_name)
  return nodes_by_device


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


class IPUMultiWorkerStrategyV1MultiProcessTest(googletest.TestCase):
  """Tests using multiple processes."""
  def _run_task_in_process(self, task_fn, cluster_spec, task_type, task_id):
    def wrapper_fn():
      os.environ["TF_CONFIG"] = json.dumps({
          "cluster": cluster_spec,
          "rpc_layer": "grpc",
          "task": {
              "type": task_type,
              "index": task_id
          }
      })
      task_fn(task_id)

    return multiprocessing.Process(target=wrapper_fn)

  def _run_workers_in_processes(self, task_fn, cluster_spec):
    task_type = "worker"

    processes = []
    for task_id in range(len(cluster_spec[task_type])):
      p = self._run_task_in_process(task_fn, cluster_spec, task_type, task_id)
      p.start()
      processes.append(p)

    # Join all the processes before asserting to avoid any orphans.
    for p in processes:
      p.join()

    for p in processes:
      self.assertEqual(0, p.exitcode)

  def _create_test_objects(self, start_server=True, variables_on_host=True):
    cluster_resolver = TFConfigClusterResolver()
    strategy = IPUMultiWorkerStrategyV1(cluster_resolver,
                                        variables_on_host=variables_on_host)

    sess_config = config_pb2.ConfigProto()
    sess_config.allow_soft_placement = False
    sess_config.log_device_placement = False
    # The line below sets `sess_config.experimental.collective_group_leader`
    sess_config = strategy.update_config_proto(sess_config)

    if start_server:
      server = server_lib.Server(cluster_resolver.cluster_spec(),
                                 job_name=cluster_resolver.task_type,
                                 task_index=cluster_resolver.task_id,
                                 protocol=cluster_resolver.rpc_layer,
                                 config=sess_config)
      target = server.target
    else:
      target = None

    return strategy, target, sess_config

  def _test_reduction_in_compiled_cluster(self, task_id):
    strategy, target, sess_config = self._create_test_objects()

    with strategy.scope():

      def device_fn(x):
        y = x * x
        # Test both without and with an explicit outside_compilation_scope.
        sum_y = strategy.reduce(ReduceOp.SUM, y, axis=None)
        z = sum_y * sum_y
        with scopes.outside_compilation_scope():
          sum_z = strategy.reduce(ReduceOp.SUM, z, axis=None)
        return sum_z

      inputs = array_ops.placeholder(dtype=np.float32, shape=())
      with ipu_scope("/device:IPU:0"):
        compiled_fn = ipu_compiler.compile(device_fn, inputs=[inputs])

      config = IPUConfig()
      config.auto_select_ipus = 1
      tu.add_hw_ci_connection_options(config)
      config.configure_ipu_system()

      with session_lib.Session(target=target, config=sess_config) as sess:
        [out] = sess.run(compiled_fn, feed_dict={inputs: task_id + 1})
        self.assertEqual(out, 50.0)  # 2 * (1^2 + 2^2)^2

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_reduction_in_compiled_cluster(self):
    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    self._run_workers_in_processes(self._test_reduction_in_compiled_cluster,
                                   cluster_spec)

  def _test_optimizer_in_compiled_cluster(self, task_id):
    strategy, target, sess_config = self._create_test_objects(
        variables_on_host=False)

    per_worker_x = [i + 1.0 for i in range(strategy.num_replicas_in_sync)]
    x = per_worker_x[task_id]
    initial_w = 2.0
    learning_rate = 0.5

    with strategy.scope():

      def device_fn(features):
        w = variable_scope.get_variable(name="w", initializer=initial_w)
        loss = w * features
        optimizer = GradientDescentOptimizer(learning_rate)
        return optimizer.minimize(loss)

      def compiled_fn():
        return ipu_compiler.compile(device_fn, inputs=[x])

      train_op = strategy.run(compiled_fn, args=[])

      config = IPUConfig()
      config.auto_select_ipus = 1
      tu.add_hw_ci_connection_options(config)
      config.configure_ipu_system()

      [w] = variables.global_variables()

      with session_lib.Session(target=target, config=sess_config) as sess:
        sess.run(w.initializer)
        sess.run(train_op)
        expected_w = initial_w - learning_rate * np.sum(per_worker_x)
        self.assertEqual(expected_w, sess.run(w))

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_optimizer_in_compiled_cluster(self):
    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    self._run_workers_in_processes(self._test_optimizer_in_compiled_cluster,
                                   cluster_spec)

  def _test_pipelining(self, task_id):
    strategy, target, sess_config = self._create_test_objects(
        variables_on_host=False)

    cpu_device = "/job:worker/replica:0/task:{}/device:CPU:0".format(task_id)
    ipu_device = "/job:worker/replica:0/task:{}/device:IPU:0".format(task_id)

    per_worker_x = [i + 1.0 for i in range(strategy.num_replicas_in_sync)]
    y = 1.0
    initial_w0 = 1.0
    initial_w1 = 2.0
    learning_rate = 0.5
    gradient_accumulation_count = 4
    num_iterations = 4
    repeat_count = 2

    num_session_runs, remainder = divmod(num_iterations, repeat_count)
    self.assertEqual(remainder, 0)

    with strategy.scope():

      x = per_worker_x[task_id]
      features = [x] * num_iterations * gradient_accumulation_count
      labels = [y] * num_iterations * gradient_accumulation_count
      dataset = dataset_ops.Dataset.from_tensor_slices((features, labels))

      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def stage1(feature, label):
        w0 = variable_scope.get_variable(name="w0", initializer=initial_w0)
        self.assertIsInstance(w0, IPUMirroredVariable)
        self.assertEqual(w0.device, ipu_device)
        partial = w0 * feature
        return partial, label

      def stage2(partial, label):
        w1 = variable_scope.get_variable(name="w1", initializer=initial_w1)
        self.assertIsInstance(w1, IPUMirroredVariable)
        self.assertEqual(w1.device, ipu_device)
        prediction = partial + w1
        loss = losses.mean_squared_error(label, prediction)
        return loss

      def optimizer_function(loss):
        opt = GradientDescentOptimizer(learning_rate)
        return pipelining_ops.OptimizerFunctionOutput(opt, loss)

      def model():
        pipeline_op = pipelining_ops.pipeline(
            computational_stages=[stage1, stage2],
            gradient_accumulation_count=gradient_accumulation_count,
            repeat_count=repeat_count,
            inputs=[],
            infeed_queue=infeed_queue,
            outfeed_queue=outfeed_queue,
            optimizer_function=optimizer_function,
            name="Pipeline")
        return pipeline_op

      def compiled_model():
        return ipu_compiler.compile(model, inputs=[])

      train_op = strategy.run(compiled_model, args=[])
      config = IPUConfig()
      config.auto_select_ipus = 2
      tu.add_hw_ci_connection_options(config)
      config.configure_ipu_system()

      expected_w0 = initial_w0
      expected_w1 = initial_w1
      w0, w1 = variables.global_variables()
      self.assertEqual("w0:0", w0.name)
      self.assertEqual("w1:0", w1.name)

      with session_lib.Session(target=target, config=sess_config) as sess:
        sess.run(infeed_queue.initializer)
        sess.run(variables.global_variables_initializer())

        run_metadata = config_pb2.RunMetadata()

        for i in range(num_session_runs):
          if i < num_session_runs - 1:
            sess.run(train_op)
          else:
            # Save execution trace for the last run.
            options = config_pb2.RunOptions(
                trace_level=config_pb2.RunOptions.FULL_TRACE)
            sess.run(train_op, options=options, run_metadata=run_metadata)

          for _ in range(repeat_count):
            # L(x) = sum_i (w_0 * x_i + w_1 - y)^2

            # dL(x)/dw_0 = sum_i 2 (w_0 * x_i + w_1 - y) x_i
            grad_w0 = sum(2 * (expected_w0 * x_i + expected_w1 - y) * x_i
                          for x_i in per_worker_x)
            accumulated_grad_w0 = gradient_accumulation_count * grad_w0

            # dL(x)/dw_1 = sum_i 2 (w_0 * x_i + w_1 - y)
            grad_w1 = sum(2 * (expected_w0 * x_i + expected_w1 - y)
                          for x_i in per_worker_x)
            accumulated_grad_w1 = gradient_accumulation_count * grad_w1

            expected_w0 -= learning_rate * accumulated_grad_w0
            expected_w1 -= learning_rate * accumulated_grad_w1

          self.assertEqual(expected_w0, sess.run(w0))
          self.assertEqual(expected_w1, sess.run(w1))

      # Do some sanity checks on what actually executed the last iteration.
      nodes_by_device = _get_executed_nodes_by_device(run_metadata)
      cpu_nodes = nodes_by_device[cpu_device]
      ipu_nodes = nodes_by_device[ipu_device]

      # There should be 2 reductions on the CPU per repeat loop iteration
      # (one for each gradient).
      self.assertEqual(2 * repeat_count,
                       sum(1 for n in cpu_nodes if "CollectiveReduce" in n))

      # There should be 1 XLA run on the IPU
      self.assertEqual(1, sum(1 for n in ipu_nodes if "xla_run" in n))

  @tu.test_may_use_ipus_or_model(num_ipus=4)
  def test_pipelining(self):
    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    self._run_workers_in_processes(self._test_pipelining, cluster_spec)

  def _run_pipelining_example_with_keras_layers(self,
                                                strategy,
                                                dataset,
                                                gradient_accumulation_count,
                                                sess_target=None,
                                                sess_config=None):
    loss_vals = []

    # Start of verbatim copy of example from ipu_multi_worker_strategy.py.

    with strategy.scope():

      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def stage1(lr, images, labels):
        partial = keras.layers.Dense(256, activation="relu")(images)
        partial = keras.layers.Dense(128, activation="relu")(partial)
        return lr, partial, labels

      def stage2(lr, partial, labels):
        logits = keras.layers.Dense(10)(partial)
        per_example_loss = keras.losses.sparse_categorical_crossentropy(
            y_true=labels, y_pred=logits, from_logits=True)
        # In a custom training loop, the optimiser does an allreduce *sum*, not
        # average, of the gradients across the distributed workers. Therefore
        # we want to divide the loss here by the *global* batch size, which is
        # done by the `tf.nn.compute_average_loss()` function.
        loss = nn.compute_average_loss(per_example_loss)
        return lr, loss

      def optimizer_function(lr, loss):
        optimizer = GradientDescentOptimizer(lr)
        return pipelining_ops.OptimizerFunctionOutput(optimizer, loss)

      def model(lr):
        pipeline_op = pipelining_ops.pipeline(
            computational_stages=[stage1, stage2],
            gradient_accumulation_count=gradient_accumulation_count,
            inputs=[lr],
            infeed_queue=infeed_queue,
            outfeed_queue=outfeed_queue,
            optimizer_function=optimizer_function,
            name="Pipeline")
        return pipeline_op

      def compiled_model(lr):
        with ipu_scope("/device:IPU:0"):
          return ipu_compiler.compile(model, inputs=[lr])

      with ops.device("cpu"):
        lr = array_ops.placeholder(np.float32, [])

      train_op = strategy.run(compiled_model, args=[lr])

      _, per_worker_losses = outfeed_queue.dequeue()

      # Mean across the local `gradient_accumulation_count` batches:
      per_worker_loss = math_ops.reduce_mean(per_worker_losses)

      # Global mean across the distributed workers (since it is already
      # divided by the global batch size above, we do a sum here):
      global_loss = strategy.reduce(ReduceOp.SUM, per_worker_loss)

      config = IPUConfig()
      config.auto_select_ipus = 2
      tu.add_hw_ci_connection_options(config)
      config.configure_ipu_system()
      ipu_utils.move_variable_initialization_to_cpu()

      with session_lib.Session(target=sess_target, config=sess_config) as sess:
        sess.run(infeed_queue.initializer)
        sess.run(variables.global_variables_initializer())

        for _ in range(10):
          sess.run(train_op, {lr: 0.01})
          global_loss_val = sess.run(global_loss)

          # End of example code.

          if loss_vals:
            # Check that the loss decreases monotonically.
            self.assertLess(global_loss_val, loss_vals[-1])
          loss_vals.append(global_loss_val)

        sess.run(infeed_queue.deleter)
        sess.run(outfeed_queue.deleter)

    return loss_vals

  def _test_pipelining_example_with_keras_layers(self, task_id):
    gradient_accumulation_count = 4
    local_batch_size = 2
    num_workers = 2

    features = [
        i * np.ones((1, 20), dtype=np.float32) for i in range(num_workers)
    ]
    labels = [i * np.ones(1, dtype=np.int32) for i in range(num_workers)]
    concat_features = np.concatenate(features)
    concat_labels = np.concatenate(labels)

    def mock_initializer_get(_identifier):
      return init_ops.GlorotUniform(seed=42)

    with test.mock.patch.object(keras.initializers, 'get',
                                mock_initializer_get):

      # Test using the default non-distributed strategy. Each batch is twice
      # the size, with the batches for the workers concatenated.
      default_strategy = distribution_strategy_context._get_default_strategy()  # pylint: disable=protected-access

      with ops.Graph().as_default():
        concat_dataset = dataset_ops.Dataset.from_tensor_slices(
            (concat_features, concat_labels))
        concat_dataset = concat_dataset.repeat().batch(num_workers *
                                                       local_batch_size,
                                                       drop_remainder=True)
        losses_reference = self._run_pipelining_example_with_keras_layers(
            default_strategy, concat_dataset, gradient_accumulation_count)

      # Test using the actual distribution strategy. Each worker gets its own batch.
      strategy, sess_target, sess_config = self._create_test_objects(
          variables_on_host=False)

      with ops.Graph().as_default():
        local_dataset = dataset_ops.Dataset.from_tensor_slices(
            (features[task_id], labels[task_id]))
        local_dataset = local_dataset.repeat().batch(local_batch_size,
                                                     drop_remainder=True)
        losses_distributed = self._run_pipelining_example_with_keras_layers(
            strategy, local_dataset, gradient_accumulation_count, sess_target,
            sess_config)

    # The resulting losses should be the same, as distributed training should in
    # general be equivalent to non-distributed training with concatenated batches.
    np.testing.assert_almost_equal(losses_reference,
                                   losses_distributed,
                                   decimal=6)

  @tu.test_may_use_ipus_or_model(num_ipus=4)
  def test_pipelining_example_with_keras_layers(self):
    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    self._run_workers_in_processes(
        self._test_pipelining_example_with_keras_layers, cluster_spec)

  def _test_ipu_pipeline_estimator(self, task_id):
    # The estimator library starts the server when configured in TF_CONFIG.
    strategy, _, _ = self._create_test_objects(start_server=False,
                                               variables_on_host=False)

    ipu_device = "/job:worker/replica:0/task:{}/device:IPU:0".format(task_id)

    num_iterations = 3
    num_workers = strategy.num_replicas_in_sync
    per_worker_x = [i + 1.0 for i in range(num_workers)]
    y = 1.0
    initial_w0 = 1.0
    initial_w1 = 2.0
    learning_rate = 0.5
    gradient_accumulation_count = 4
    num_steps = num_iterations * gradient_accumulation_count

    def my_model_fn(mode):
      def stage1(feature, label):
        w0 = variable_scope.get_variable(name="w0", initializer=initial_w0)
        self.assertIsInstance(w0, IPUMirroredVariable)
        self.assertEqual(w0.device, ipu_device)
        partial = w0 * feature
        return partial, label

      def stage2(partial, label):
        w1 = variable_scope.get_variable(name="w1", initializer=initial_w1)
        self.assertIsInstance(w1, IPUMirroredVariable)
        self.assertEqual(w1.device, ipu_device)
        prediction = partial + w1
        loss = losses.mean_squared_error(label, prediction)
        return loss

      def optimizer_function(loss):
        # Use momentum, but set to zero, just to verify that the
        # momentum accumulator "slot" does not cause any problems.
        opt = MomentumOptimizer(learning_rate, momentum=0.0)
        return pipelining_ops.OptimizerFunctionOutput(opt, loss)

      return ipu_pipeline_estimator.IPUPipelineEstimatorSpec(
          mode=mode,
          computational_stages=[stage1, stage2],
          gradient_accumulation_count=gradient_accumulation_count,
          optimizer_function=optimizer_function)

    def my_input_fn(input_context):
      self.assertEqual(task_id, input_context.input_pipeline_id)

      x = per_worker_x[task_id]
      features = [x] * num_steps
      labels = [y] * num_steps
      dataset = dataset_ops.Dataset.from_tensor_slices((features, labels))

      return dataset

    num_ipus_in_pipeline = 2
    ipu_options = IPUConfig()
    ipu_options.auto_select_ipus = num_ipus_in_pipeline
    tu.add_hw_ci_connection_options(ipu_options)

    config = ipu_run_config.RunConfig(
        session_config=config_pb2.ConfigProto(allow_soft_placement=False),
        train_distribute=strategy,
        save_summary_steps=1,
        ipu_run_config=ipu_run_config.IPURunConfig(
            iterations_per_loop=gradient_accumulation_count,
            num_shards=num_ipus_in_pipeline,
            ipu_options=ipu_options))

    estimator = ipu_pipeline_estimator.IPUPipelineEstimator(
        model_fn=my_model_fn, config=config)

    estimator_lib.train_and_evaluate(
        estimator,
        train_spec=estimator_lib.TrainSpec(input_fn=my_input_fn,
                                           max_steps=num_steps),
        eval_spec=estimator_lib.EvalSpec(input_fn=my_input_fn,
                                         steps=num_steps))

    expected_w0 = initial_w0
    expected_w1 = initial_w1
    expected_losses = []

    for _ in range(num_iterations):
      x = np.array(per_worker_x)

      # The loss reduction op is decided by _get_loss_reduce_op_for_reporting()
      loss = np.sum(np.square(expected_w0 * x + expected_w1 - y))
      expected_losses.append(loss)

      grad_w0 = np.sum(2 * (expected_w0 * x + expected_w1 - y) * x)
      accumulated_grad_w0 = gradient_accumulation_count * grad_w0

      grad_w1 = np.sum(2 * (expected_w0 * x + expected_w1 - y))
      accumulated_grad_w1 = gradient_accumulation_count * grad_w1

      expected_w0 -= learning_rate * accumulated_grad_w0 / num_workers
      expected_w1 -= learning_rate * accumulated_grad_w1 / num_workers

    # Only the chief worker has the checkpoint to read the variables from.
    if task_id == 0:
      self.assertEqual(num_steps, estimator.get_variable_value("global_step"))
      self.assertEqual(expected_w0, estimator.get_variable_value("w0"))
      self.assertEqual(expected_w1, estimator.get_variable_value("w1"))

      loss_outputs = _get_summary_values(estimator.model_dir, "loss")
      self.assertEqual(expected_losses, loss_outputs)

  @tu.test_may_use_ipus_or_model(num_ipus=4)
  def test_ipu_pipeline_estimator(self):
    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    self._run_workers_in_processes(self._test_ipu_pipeline_estimator,
                                   cluster_spec)

  def _test_dataset_infeed(self, task_id):
    strategy, target, sess_config = self._create_test_objects()

    with strategy.scope():
      dataset = dataset_ops.Dataset.from_tensor_slices([0.0]).repeat()
      # Test with a dataset host op.
      dataset = dataset.map(lambda x: x + task_id)
      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)

      def body(v, x):
        v += x
        return v

      def my_net():
        r = loops.repeat(10, body, [0.0], infeed_queue)
        return r

      with ipu_scope("/device:IPU:0"):
        [res] = ipu_compiler.compile(my_net, inputs=[])

      config = IPUConfig()
      config.auto_select_ipus = 1
      tu.add_hw_ci_connection_options(config)
      config.configure_ipu_system()

      with session_lib.Session(target=target, config=sess_config) as sess:
        sess.run(infeed_queue.initializer)
        self.assertEqual(task_id * 10.0, sess.run(res))

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_dataset_infeed(self):
    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    self._run_workers_in_processes(self._test_dataset_infeed, cluster_spec)

  def _test_ipu_estimator(self, task_id):
    # The estimator library starts the server when configured in TF_CONFIG.
    strategy, _, _ = self._create_test_objects(start_server=False,
                                               variables_on_host=False)

    ipu_device = "/job:worker/replica:0/task:{}/device:IPU:0".format(task_id)

    num_iterations = 3
    num_workers = strategy.num_replicas_in_sync
    per_worker_x = [i + 1.0 for i in range(num_workers)]
    y = 1.0
    initial_w0 = 1.0
    initial_w1 = 2.0
    learning_rate = 0.5

    def my_model_fn(features, labels, mode):
      w0 = variable_scope.get_variable(name="w0", initializer=initial_w0)
      self.assertIsInstance(w0, IPUMirroredVariable)
      self.assertEqual(w0.device, ipu_device)
      partial = w0 * features

      w1 = variable_scope.get_variable(name="w1", initializer=initial_w1)
      self.assertIsInstance(w1, IPUMirroredVariable)
      self.assertEqual(w1.device, ipu_device)
      prediction = partial + w1
      loss = losses.mean_squared_error(labels, prediction)

      # Use momentum, but set to zero, just to verify that the
      # momentum accumulator "slot" does not cause any problems.
      opt = MomentumOptimizer(learning_rate, momentum=0.0)
      train_op = opt.minimize(loss)

      return ipu_estimator.IPUEstimatorSpec(mode=mode,
                                            loss=loss,
                                            train_op=train_op)

    def my_input_fn(input_context):
      self.assertEqual(task_id, input_context.input_pipeline_id)

      x = per_worker_x[task_id]
      features = [x] * num_iterations
      labels = [y] * num_iterations
      dataset = dataset_ops.Dataset.from_tensor_slices((features, labels))

      return dataset

    ipu_options = IPUConfig()
    ipu_options.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(ipu_options)

    config = ipu_run_config.RunConfig(
        session_config=config_pb2.ConfigProto(allow_soft_placement=False),
        train_distribute=strategy,
        save_summary_steps=1,
        ipu_run_config=ipu_run_config.IPURunConfig(ipu_options=ipu_options))

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)

    estimator_lib.train_and_evaluate(
        estimator,
        train_spec=estimator_lib.TrainSpec(input_fn=my_input_fn,
                                           max_steps=num_iterations),
        eval_spec=estimator_lib.EvalSpec(input_fn=my_input_fn))

    expected_w0 = initial_w0
    expected_w1 = initial_w1
    expected_losses = []

    for _ in range(num_iterations):
      x = np.array(per_worker_x)

      # The loss reduction op is decided by _get_loss_reduce_op_for_reporting()
      loss = np.mean(np.square(expected_w0 * x + expected_w1 - y))
      expected_losses.append(loss)

      grad_w0 = np.sum(2 * (expected_w0 * x + expected_w1 - y) * x)
      grad_w1 = np.sum(2 * (expected_w0 * x + expected_w1 - y))

      expected_w0 -= learning_rate * grad_w0 / num_workers
      expected_w1 -= learning_rate * grad_w1 / num_workers

    # Only the chief worker has the checkpoint to read the variables from.
    if task_id == 0:
      self.assertEqual(num_iterations,
                       estimator.get_variable_value("global_step"))
      self.assertEqual(expected_w0, estimator.get_variable_value("w0"))
      self.assertEqual(expected_w1, estimator.get_variable_value("w1"))

      loss_outputs = _get_summary_values(estimator.model_dir, "loss")
      self.assertEqual(expected_losses, loss_outputs)

  @tu.test_may_use_ipus_or_model(num_ipus=2)
  def test_ipu_estimator(self):
    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    self._run_workers_in_processes(self._test_ipu_estimator, cluster_spec)


if __name__ == "__main__":
  test.main()
