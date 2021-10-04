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

from tensorflow import debugging
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute.reduce_util import ReduceOp
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu.horovod import ipu_multi_replica_strategy
from tensorflow.python.ipu.ipu_multi_worker_strategy import IPUSyncOnReadVariable
from tensorflow.python.ops import variables
from tensorflow.python.ops.variable_scope import VariableAggregation, VariableSynchronization
from tensorflow.python.platform import test


class IPUMultiReplicaStrategyV1Test(test_util.TensorFlowTestCase):  # pylint: disable=abstract-method
  @classmethod
  def setUpClass(cls):
    hvd.init()

  @classmethod
  def tearDownClass(cls):
    hvd.shutdown()

  def test_update_ipu_config(self):
    strategy = ipu_multi_replica_strategy.IPUMultiReplicaStrategyV1()
    config = IPUConfig()
    strategy.update_ipu_config(config)
    self.assertEqual(
        config.experimental.multi_replica_distribution.process_count,
        hvd.size())
    self.assertEqual(
        config.experimental.multi_replica_distribution.process_index,
        hvd.rank())

  def test_strategy(self):
    config = IPUConfig()
    config.auto_select_ipus = 1
    config.configure_ipu_system()

    hvd.init()

    strategy = ipu_multi_replica_strategy.IPUMultiReplicaStrategyV1()

    with strategy.scope():
      v = variables.Variable(initial_value=hvd.rank() + 1, dtype=np.float32)
      self.assertEndsWith(v.device, "/device:IPU:0")

      @def_function.function
      def per_replica_fn(x):
        y = v * x

        replica_context = distribution_strategy_context.get_replica_context()

        # This reduction is done on IPU, and hence uses GCL. In this case,
        # since there is no replication in this test, it is an identity op.
        # We cannot explicitly check for the device of the result, as the
        # @tf.function decorator does not specify this anymore.
        y_all_reduced = replica_context.all_reduce(ReduceOp.SUM, y)

        # Sanity check that replication normalise does not support int.
        with self.assertRaisesRegex(TypeError,
                                    "int32 not in list of allowed values"):
          replica_context.all_reduce(ReduceOp.MEAN, 1)

        return y_all_reduced

      per_replica_value = strategy.run(per_replica_fn,
                                       args=[constant_op.constant(2.0)])

      # This reduction is performed on CPU, and hence uses Horovod.
      value_all_reduced = strategy.reduce(ReduceOp.SUM, per_replica_value)

      # The initial value should be broadcast from rank 0.
      self.assertEqual(v, 1.0)

      # There should be one allreduce sum of the values.
      self.assertEqual(value_all_reduced, hvd.size() * 2.0)

  def test_strategy_without_ipu_reduction(self):
    config = IPUConfig()
    config.auto_select_ipus = 1
    config.configure_ipu_system()

    hvd.init()

    strategy = ipu_multi_replica_strategy.IPUMultiReplicaStrategyV1(
        add_ipu_cross_replica_reductions=False)

    with strategy.scope():
      v = variables.Variable(initial_value=1.0, dtype=np.float32)

      @def_function.function
      def per_replica_fn(x):
        y = v * x
        replica_context = distribution_strategy_context.get_replica_context()

        # Since IPU reductions are disabled, this should be an identity op.
        y_out = replica_context.all_reduce(ReduceOp.SUM, y)
        debugging.assert_equal(y_out.op.type, "IdentityN")
        debugging.assert_equal(y_out.op.inputs[0], y)
        return y_out

      # It is sufficient to test the TF graph construction.
      strategy.run(per_replica_fn, args=[constant_op.constant(2.0)])

  def test_strategy_with_sync_on_read_variable(self):
    config = IPUConfig()
    config.auto_select_ipus = 1
    config.configure_ipu_system()

    hvd.init()

    strategy = ipu_multi_replica_strategy.IPUMultiReplicaStrategyV1()

    with strategy.scope():
      w = variables.Variable(initial_value=float(hvd.rank() + 1),
                             dtype=np.float32,
                             synchronization=VariableSynchronization.ON_READ,
                             aggregation=VariableAggregation.MEAN)

      @def_function.function
      def per_replica_fn(x):
        self.assertIsInstance(w, IPUSyncOnReadVariable)
        w.assign(x + w)

        return w

      # Both should have initial value from first worker
      debugging.assert_equal([1.0], w)
      strategy.run(per_replica_fn,
                   args=[constant_op.constant(hvd.rank() + 1.0)])
      debugging.assert_equal([2.5], w.read_value())


if __name__ == "__main__":
  test.main()
