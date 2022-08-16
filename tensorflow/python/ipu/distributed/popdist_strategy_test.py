# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
import numpy as np
import popdist

from tensorflow.python import ipu
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute.reduce_util import ReduceOp
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu.distributed import popdist_strategy
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.monitored_session import MonitoredTrainingSession


class PopDistStrategyTest(test_util.TensorFlowTestCase):  # pylint: disable=abstract-method
  @classmethod
  def setUpClass(cls):
    hvd.init()
    popdist.init()

  @classmethod
  def tearDownClass(cls):
    hvd.shutdown()

  def test_update_ipu_config(self):
    strategy = popdist_strategy.PopDistStrategy()
    config = ipu.config.IPUConfig()
    strategy.update_ipu_config(config)
    self.assertEqual(
        config.experimental.multi_replica_distribution.process_count,
        popdist.getNumInstances())
    self.assertEqual(
        config.experimental.multi_replica_distribution.process_index,
        popdist.getInstanceIndex())

  @test_util.deprecated_graph_mode_only
  def test_strategy(self):
    strategy = popdist_strategy.PopDistStrategy()

    with strategy.scope():

      v = variables.Variable(initial_value=popdist.getInstanceIndex() + 1,
                             dtype=np.float32)
      self.assertEndsWith(v.device, "/device:IPU:0")

      def per_replica_fn(x):
        y = v * x

        replica_context = distribution_strategy_context.get_replica_context()

        # This reduction is done on IPU, and hence uses GCL. In this case,
        # since there is no replication in this test, it is an identity op.
        y_allreduced = replica_context.all_reduce(ReduceOp.SUM, y)
        self.assertEndsWith(y_allreduced.device, "/device:IPU:0")

        # Sanity check that replication normalise does not support int.
        with self.assertRaisesRegex(TypeError,
                                    "int32 not in list of allowed values"):
          replica_context.all_reduce(ReduceOp.MEAN, 1)

        return y_allreduced

      per_replica_value = strategy.experimental_run_v2(
          per_replica_fn, args=[constant_op.constant(2.0)])

      # This reduction is performed on CPU, and hence uses Horovod.
      value_allreduced = strategy.reduce(ReduceOp.SUM, per_replica_value)

      with session.Session() as sess:
        config = ipu.config.IPUConfig()
        config.auto_select_ipus = 1
        config.configure_ipu_system()

        sess.run(v.initializer)

        # The initial value should be broadcast from rank 0.
        self.assertEqual(sess.run(v), 1.0)

        # There should be one allreduce sum of the values.
        self.assertEqual(sess.run(value_allreduced),
                         popdist.getNumInstances() * 2.0)

  @test_util.deprecated_graph_mode_only
  def test_strategy_without_ipu_reduction(self):
    strategy = popdist_strategy.PopDistStrategy(
        add_ipu_cross_replica_reductions=False)

    with strategy.scope():

      v = variables.Variable(initial_value=1.0, dtype=np.float32)

      def per_replica_fn(x):
        y = v * x

        replica_context = distribution_strategy_context.get_replica_context()

        # Since IPU reductions are disabled, this should be an identity op.
        y_out = replica_context.all_reduce(ReduceOp.SUM, y)
        self.assertEqual(y_out.op.type, "IdentityN")
        self.assertEqual(y_out.op.inputs[0], y)
        return y_out

      # It is sufficient to test the TF graph construction.
      strategy.experimental_run_v2(per_replica_fn,
                                   args=[constant_op.constant(2.0)])

  @test_util.deprecated_graph_mode_only
  def test_strategy_with_sync_on_read_variable(self):
    strategy = popdist_strategy.PopDistStrategy()

    with strategy.scope():

      def per_replica_fn(x):
        w0 = variable_scope.get_variable(
            name="w0",
            initializer=float(popdist.getInstanceIndex() + 1),
            synchronization=variable_scope.VariableSynchronization.ON_READ,
            aggregation=variable_scope.VariableAggregation.MEAN)
        self.assertIsInstance(w0, popdist_strategy.IPUSyncOnReadVariable)
        return w0.assign_add(x)

      inputs = array_ops.placeholder(dtype=np.float32, shape=())
      assign_add_op = strategy.experimental_run_v2(per_replica_fn,
                                                   args=[inputs])

      with session.Session() as sess:
        config = ipu.config.IPUConfig()
        strategy.update_ipu_config(config)
        config.configure_ipu_system()
        sess.run(variables.global_variables_initializer())
        # Both should have initial value from first worker
        self.assertEqual([1.0], sess.run(variables.global_variables()))
        sess.run(assign_add_op,
                 feed_dict={inputs: popdist.getInstanceIndex() + 1})
        # mean(1 + 1, 1 + 2) = 2.5
        self.assertEqual([2.5], sess.run(variables.global_variables()))

  @test_util.deprecated_graph_mode_only
  def test_all_reduce(self):
    strategy = popdist_strategy.PopDistStrategy()

    with strategy.scope():

      def per_replica_fn(x):
        return x * x

      inputs = array_ops.placeholder(dtype=np.float32, shape=())
      per_replica_y = strategy.experimental_run_v2(per_replica_fn,
                                                   args=[inputs])
      sum_y = strategy.reduce(ReduceOp.SUM, per_replica_y, axis=None)

      with session.Session() as sess:
        config = ipu.config.IPUConfig()
        config.auto_select_ipus = 1
        config.configure_ipu_system()
        sess.run(variables.global_variables_initializer())
        out = sess.run(sum_y,
                       feed_dict={inputs: popdist.getInstanceIndex() + 1})
        self.assertEqual(5.0, out)  # 1*1 + 2*2

  @test_util.deprecated_graph_mode_only
  def test_mirrored_variable(self):
    strategy = popdist_strategy.PopDistStrategy()

    with strategy.scope():

      def per_replica_fn():
        with ops.device("/device:IPU:0"):
          w0 = variable_scope.get_variable(
              name="w0", initializer=popdist.getInstanceIndex() + 1)
          self.assertIsInstance(w0, popdist_strategy.IPUMirroredVariable)
          ret = w0 * w0
          return ret

      per_replica_ret = strategy.experimental_run_v2(per_replica_fn, args=[])
      sum_ret = strategy.reduce(ReduceOp.SUM, per_replica_ret, axis=None)

      with session.Session() as sess:
        config = ipu.config.IPUConfig()
        config.auto_select_ipus = 1
        config.configure_ipu_system()
        sess.run(variables.global_variables_initializer())
        # Both should have initial value from first worker
        self.assertEqual([1.0], sess.run(variables.global_variables()))
        self.assertEqual(2.0, sess.run(sum_ret))  # 1*1 + 1*1

  @test_util.deprecated_graph_mode_only
  def test_monitored_training_session(self):
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.configure_ipu_system()

    strategy = popdist_strategy.PopDistStrategy()

    with strategy.scope():

      def step_fn(x):
        w = variable_scope.get_variable("w", initializer=2.0)
        y = w * x
        return y

      inputs = array_ops.placeholder(dtype=np.float32, shape=())
      per_replica_y = strategy.experimental_run_v2(step_fn, args=[inputs])
      sum_y = strategy.reduce(ReduceOp.SUM, per_replica_y, axis=None)

      with MonitoredTrainingSession() as sess:
        out = sess.run(sum_y,
                       feed_dict={inputs: popdist.getInstanceIndex() + 1})
        self.assertEqual(6.0, out)  # 2*1 + 2*2

  @test_util.deprecated_graph_mode_only
  def test_batch_normalization(self):
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.configure_ipu_system()

    strategy = popdist_strategy.PopDistStrategy()

    with strategy.scope():
      batch_norm = BatchNormalization(momentum=0.0)

      def per_replica_fn(x):
        y = batch_norm(x, training=True)

        self.assertIsInstance(batch_norm.beta,
                              popdist_strategy.IPUMirroredVariable)
        self.assertIsInstance(batch_norm.gamma,
                              popdist_strategy.IPUMirroredVariable)

        self.assertIsInstance(batch_norm.moving_mean,
                              popdist_strategy.IPUSyncOnReadVariable)
        self.assertIsInstance(batch_norm.moving_variance,
                              popdist_strategy.IPUSyncOnReadVariable)

        return y

      def compiled_per_replica_fn(inputs):
        [out] = ipu_compiler.compile(per_replica_fn, inputs=[inputs])
        return out

      inputs = array_ops.placeholder(dtype=np.float32, shape=(2, 1))
      per_replica_y = strategy.experimental_run_v2(compiled_per_replica_fn,
                                                   args=[inputs])
      sum_y = strategy.reduce(ReduceOp.SUM, per_replica_y, axis=None)

      with session.Session() as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(sum_y,
                 feed_dict={
                     inputs: [[2.0 * (popdist.getInstanceIndex() + 1)], [0.0]]
                 })

        # mean(mean(2, 0), mean(4, 0)) = mean(1, 3) = 1.5
        global_mean = batch_norm.moving_mean
        self.assertAllEqual([1.5], sess.run(global_mean))

        # mean(var(2, 0), var(4, 0)) = mean(1, 4) = 2.5
        self.assertAllEqual([2.5], sess.run(batch_norm.moving_variance))

  @test_util.deprecated_graph_mode_only
  def test_dataset_infeed(self):
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.configure_ipu_system()
    strategy = popdist_strategy.PopDistStrategy()

    with strategy.scope():
      dataset = dataset_ops.Dataset.from_tensor_slices([0.0]).repeat()
      # Test with a dataset host op.
      dataset = dataset.map(lambda x: x + popdist.getInstanceIndex())
      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)

      def body(v, x):
        v += x
        return v

      def my_net():
        r = loops.repeat(10, body, [0.0], infeed_queue)
        return r

      with ipu_scope("/device:IPU:0"):
        [res] = ipu_compiler.compile(my_net, inputs=[])

      with session.Session() as sess:
        sess.run(infeed_queue.initializer)
        self.assertEqual(popdist.getInstanceIndex() * 10.0, sess.run(res))


if __name__ == "__main__":
  test.main()
