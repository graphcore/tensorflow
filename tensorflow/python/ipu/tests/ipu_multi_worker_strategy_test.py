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

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.reduce_util import ReduceOp
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_estimator
from tensorflow.python.ipu import ipu_run_config
from tensorflow.python.ipu.ipu_multi_worker_strategy import IPUMultiWorkerStrategy
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.python.training.momentum import MomentumOptimizer
from tensorflow.python.training.monitored_session import MonitoredTrainingSession


class IPUMultiWorkerStrategyTest(multi_worker_test_base.MultiWorkerTestBase):
  @classmethod
  def setUpClass(cls):
    cls._num_workers = 2
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=cls._num_workers, num_ps=0, has_chief=False)

  collective_key_base = 0

  def setUp(self):
    # We use a different key_base for each test so that collective keys won't be
    # reused.
    IPUMultiWorkerStrategyTest.collective_key_base += 100000
    super(IPUMultiWorkerStrategyTest, self).setUp()

  def _create_test_objects(self, task_type, task_id):
    sess_config = config_pb2.ConfigProto()
    sess_config.allow_soft_placement = False
    sess_config.log_device_placement = False

    cluster_spec = multi_worker_util.normalize_cluster_spec(self._cluster_spec)
    cluster_resolver = SimpleClusterResolver(
        cluster_spec=cluster_spec,
        task_type=task_type,
        task_id=task_id)
    target = cluster_resolver.master(task_id=task_id, task_type=task_type, rpc_layer="grpc")
    strategy = IPUMultiWorkerStrategy(cluster_resolver)
    sess_config = strategy.update_config_proto(sess_config)

    collective_keys = cross_device_utils.CollectiveKeys(
        group_key_start=10 + IPUMultiWorkerStrategyTest.collective_key_base,
        op_instance_key_start=100 + IPUMultiWorkerStrategyTest.collective_key_base,
        variable_instance_key_start=10000 + IPUMultiWorkerStrategyTest.collective_key_base)
    strategy.extended._collective_keys = collective_keys
    strategy.extended._cross_device_ops._collective_keys = collective_keys

    return strategy, target, sess_config

  def test_strategy_first_worker(self):
    strategy, _, _ = self._create_test_objects(task_type="worker", task_id=0)
    self.assertEqual(2, strategy.num_replicas_in_sync)
    self.assertEqual(True, strategy.extended.experimental_between_graph)
    self.assertEqual(True, strategy.extended.experimental_should_init)
    self.assertEqual(True, strategy.extended.should_checkpoint)
    self.assertEqual(True, strategy.extended.should_save_summary)

  def test_strategy_second_worker(self):
    strategy, _, _ = self._create_test_objects(task_type="worker", task_id=1)
    self.assertEqual(2, strategy.num_replicas_in_sync)
    self.assertEqual(True, strategy.extended.experimental_between_graph)
    self.assertEqual(True, strategy.extended.experimental_should_init)
    self.assertEqual(False, strategy.extended.should_checkpoint)
    self.assertEqual(False, strategy.extended.should_save_summary)

  def _test_all_reduce(self, task_type, task_id, _num_gpus):
    strategy, target, sess_config = self._create_test_objects(
        task_type=task_type, task_id=task_id)

    variable_device = "/job:{}/replica:0/task:{}/device:CPU:0".format(
        task_type, task_id)
    compute_device = "/job:{}/replica:0/task:{}/device:IPU:0".format(
        task_type, task_id)

    with strategy.scope():
      def per_replica_fn(x):
        with ops.device("/device:IPU:0"):
          y = x * x
          self.assertEqual(compute_device, y.device)
          return y

      inputs = array_ops.placeholder(dtype=np.float32, shape=())
      per_replica_y = strategy.experimental_run_v2(per_replica_fn, args=[inputs])
      self.assertEqual(compute_device, per_replica_y.device)
      sum_y = strategy.reduce(ReduceOp.SUM, per_replica_y, axis=None)
      self.assertEqual(variable_device, sum_y.device)

      with session_lib.Session(target=target, config=sess_config) as sess:
        out = sess.run(sum_y, feed_dict={inputs: task_id + 1})
        self.assertEqual(5.0, out)  # 1*1 + 2*2

  def test_all_reduce(self):
    self._run_between_graph_clients(self._test_all_reduce,
                                    self._cluster_spec, num_gpus=0)

  def _test_variable_placement_and_initialization(self, task_type, task_id, _num_gpus):
    strategy, target, sess_config = self._create_test_objects(
        task_type=task_type, task_id=task_id)

    variable_device = "/job:{}/replica:0/task:{}/device:CPU:0".format(
        task_type, task_id)
    compute_device = "/job:{}/replica:0/task:{}/device:IPU:0".format(
        task_type, task_id)

    with strategy.scope():
      def per_replica_fn():
        with ops.device("/device:IPU:0"):
          w0 = variable_scope.get_variable(name="w0", initializer=task_id + 1)
          self.assertEqual(variable_device, w0.device)
          cached_value = w0.value()
          self.assertEqual(compute_device, cached_value.device)
          ret = w0 * w0
          self.assertEqual(compute_device, ret.device)
          return ret

      per_replica_ret = strategy.experimental_run_v2(per_replica_fn, args=[])
      self.assertEqual(compute_device, per_replica_ret.device)
      sum_ret = strategy.reduce(ReduceOp.SUM, per_replica_ret, axis=None)
      self.assertEqual(variable_device, sum_ret.device)

      with session_lib.Session(target=target, config=sess_config) as sess:
        sess.run(variables.global_variables_initializer())
        # Both should have initial value from first worker
        self.assertEqual([1.0], sess.run(variables.global_variables()))
        self.assertEqual(2.0, sess.run(sum_ret))  # 1*1 + 1*1

  def test_variable_placement_and_initialization(self):
    self._run_between_graph_clients(self._test_variable_placement_and_initialization,
                                    self._cluster_spec, num_gpus=0)

  def _test_train_split_device_host_fn(self, task_type, task_id, _num_gpus):
    strategy, target, sess_config = self._create_test_objects(
        task_type=task_type, task_id=task_id)

    variable_device = "/job:{}/replica:0/task:{}/device:CPU:0".format(
        task_type, task_id)
    compute_device = "/job:{}/replica:0/task:{}/device:IPU:0".format(
        task_type, task_id)

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
          return train_op

      def step_fn(inputs):
        grads, loss = compiled_device_step_fn(inputs)
        train_op = host_step_fn(grads)
        return train_op, loss

      inputs = array_ops.placeholder(dtype=np.float32, shape=())
      train_op, per_replica_loss = strategy.experimental_run_v2(step_fn, args=[inputs])
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

  def test_train_split_device_host_fn(self):
    self._run_between_graph_clients(self._test_train_split_device_host_fn,
                                    self._cluster_spec, num_gpus=0)

  def _test_train_combined_device_host_fn(self, task_type, task_id, _num_gpus):
    strategy, target, sess_config = self._create_test_objects(
        task_type=task_type, task_id=task_id)

    variable_device = "/job:{}/replica:0/task:{}/device:CPU:0".format(
        task_type, task_id)
    compute_device = "/job:{}/replica:0/task:{}/device:IPU:0".format(
        task_type, task_id)

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
      train_op, per_replica_loss = strategy.experimental_run_v2(step_fn, args=[inputs])
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

  def test_train_combined_device_host_fn(self):
    self._run_between_graph_clients(self._test_train_combined_device_host_fn,
                                    self._cluster_spec, num_gpus=0)

  def _test_slot_variable_placement(self, task_type, task_id, _num_gpus):
    strategy, target, sess_config = self._create_test_objects(
        task_type=task_type, task_id=task_id)

    variable_device = "/job:{}/replica:0/task:{}/device:CPU:0".format(
        task_type, task_id)

    with strategy.scope():
      optimizer = MomentumOptimizer(learning_rate=0.5, momentum=0.9)

      def step_fn(x):
        with ipu_scope("/device:IPU:0"):
          w = variable_scope.get_variable(name="w", initializer=1.0)
          loss = w * x
          train_op = optimizer.minimize(loss)
          return train_op, loss

      inputs = array_ops.placeholder(dtype=np.float32, shape=())
      train_op, per_replica_loss = strategy.experimental_run_v2(step_fn, args=[inputs])
      total_loss = strategy.reduce(ReduceOp.SUM, per_replica_loss, axis=None)

      # Verify device placement of momentum accumulator variable.
      self.assertEqual(1, len(optimizer.variables()))
      self.assertEqual(variable_device, optimizer.variables()[0].device)

      with session_lib.Session(target=target, config=sess_config) as sess:
        sess.run(variables.global_variables_initializer())
        _, loss_val = sess.run([train_op, total_loss], feed_dict={inputs: 1.0})
        self.assertEqual(2.0, loss_val)

  def test_slot_variable_placement(self):
    self._run_between_graph_clients(self._test_slot_variable_placement,
                                    self._cluster_spec, num_gpus=0)

  def _test_distribute_dataset(self, task_type, task_id, _num_gpus):
    strategy, target, sess_config = self._create_test_objects(
        task_type=task_type, task_id=task_id)

    with strategy.scope():
      def step_fn(x):
        with ipu_scope("/device:IPU:0"):
          y = x * x
          return y

      dataset = dataset_ops.Dataset.range(10)
      dataset = dataset.batch(1, drop_remainder=True)
      dist_dataset = strategy.experimental_distribute_dataset(dataset)
      inputs = dist_dataset.make_initializable_iterator()
      per_replica_y = strategy.experimental_run_v2(step_fn, args=[next(inputs)])
      sum_y = strategy.reduce(ReduceOp.SUM, per_replica_y, axis=None)

      with session_lib.Session(target=target, config=sess_config) as sess:
        sess.run(inputs.initializer)
        self.assertEqual(1.0, sess.run(sum_y))  # 0*0 + 1*1
        self.assertEqual(13.0, sess.run(sum_y))  # 2*2 + 3*3
        self.assertEqual(41.0, sess.run(sum_y))  # 4*4 + 5*5

  def test_distribute_dataset(self):
    self._run_between_graph_clients(self._test_distribute_dataset,
                                    self._cluster_spec, num_gpus=0)

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
      per_replica_y = strategy.experimental_run_v2(step_fn, args=[inputs])
      sum_y = strategy.reduce(ReduceOp.SUM, per_replica_y, axis=None)

      with MonitoredTrainingSession(master=target, config=sess_config) as sess:
        out = sess.run(sum_y, feed_dict={inputs: task_id + 1})
        self.assertEqual(6.0, out)  # 2*1 + 2*2

  def test_monitored_training_session(self):
    self._run_between_graph_clients(self._test_monitored_training_session,
                                    self._cluster_spec, num_gpus=0)

  def _test_ipu_estimator_train(self, task_type, task_id, _num_gpus):
    strategy, target, _ = self._create_test_objects(
        task_type=task_type, task_id=task_id)

    def my_model_fn(features, labels, mode):
      loss = math_ops.reduce_sum(features + labels, name="loss")
      train_op = array_ops.identity(loss)
      return model_fn_lib.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op)

    def my_input_fn():
      features = np.array([[1.0], [2.0]], dtype=np.float32)
      labels = np.array([[3.0], [4.0]], dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((features, labels))
      dataset = dataset.batch(1, drop_remainder=True)
      dataset = dataset.shard(self._num_workers, task_id)
      return dataset

    config = ipu_run_config.RunConfig(
      master=target,
      train_distribute=strategy,
    )

    estimator = ipu_estimator.IPUEstimator(model_fn=my_model_fn, config=config)
    estimator.train(my_input_fn, steps=1)
    self.assertEquals(1, estimator.get_variable_value("global_step"))

  def test_ipu_estimator_train(self):
    self._run_between_graph_clients(self._test_ipu_estimator_train,
                                    self._cluster_spec, num_gpus=0)


if __name__ == "__main__":
  googletest.main()
