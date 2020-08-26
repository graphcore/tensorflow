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

from tensorflow.python import ipu
from tensorflow.python.client import session
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute.reduce_util import ReduceOp
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu.horovod import ipu_multi_replica_strategy
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class IPUMultiReplicaStrategyTest(test_util.TensorFlowTestCase):  # pylint: disable=abstract-method
  @classmethod
  def setUpClass(cls):
    hvd.init()

  @classmethod
  def tearDownClass(cls):
    hvd.shutdown()

  def test_update_ipu_config(self):
    strategy = ipu_multi_replica_strategy.IPUMultiReplicaStrategy()
    config = ipu.utils.create_ipu_config()
    config = strategy.update_ipu_config(config)
    self.assertEqual(config.multi_replica_process_count, hvd.size())
    self.assertEqual(config.multi_replica_process_index, hvd.rank())

  @test_util.deprecated_graph_mode_only
  def test_strategy(self):
    strategy = ipu_multi_replica_strategy.IPUMultiReplicaStrategy()

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
        config = ipu.utils.create_ipu_config()
        config = ipu.utils.auto_select_ipus(config, 1)
        ipu.utils.configure_ipu_system(config)

        sess.run(v.initializer)

        # The initial value should be broadcast from rank 0.
        self.assertEqual(sess.run(v), 1.0)

        # There should be one allreduce sum of the values.
        self.assertEqual(sess.run(value_allreduced), hvd.size() * 2.0)


if __name__ == "__main__":
  test.main()
