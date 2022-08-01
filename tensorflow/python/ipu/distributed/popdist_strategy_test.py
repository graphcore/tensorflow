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
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute.reduce_util import ReduceOp
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu.horovod import popdist_strategy
from tensorflow.python.ipu.ipu_multi_worker_strategy import IPUSyncOnReadVariable
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class IPUMultiReplicaStrategyTest(test_util.TensorFlowTestCase):  # pylint: disable=abstract-method
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
        hvd.size())
    self.assertEqual(
        config.experimental.multi_replica_distribution.process_index,
        hvd.rank())

  @test_util.deprecated_graph_mode_only
  def test_strategy(self):
    strategy = popdist_strategy.PopDistStrategy()

    with strategy.scope():

      v = variables.Variable(initial_value=hvd.rank() + 1, dtype=np.float32)
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
        self.assertEqual(sess.run(value_allreduced), hvd.size() * 2.0)

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
            initializer=float(hvd.rank() + 1),
            synchronization=variable_scope.VariableSynchronization.ON_READ,
            aggregation=variable_scope.VariableAggregation.MEAN)
        self.assertIsInstance(w0, IPUSyncOnReadVariable)
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
        sess.run(assign_add_op, feed_dict={inputs: hvd.rank() + 1})
        # mean(1 + 1, 1 + 2) = 2.5
        self.assertEqual([2.5], sess.run(variables.global_variables()))


if __name__ == "__main__":
  test.main()
