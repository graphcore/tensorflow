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
# =============================================================================
import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import def_function
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu.ops import replication_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class IPUStrategyReplicatedTest(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=2)
  @test_util.run_v2_only
  def test_all_reduce(self):
    strategy = ipu_strategy.IPUStrategy()

    def make_all_reduce_function(reduce_op):
      @def_function.function(experimental_compile=True)
      def all_reduce_function():
        replica_ctx = distribution_strategy_context.get_replica_context()
        x = math_ops.cast(replication_ops.replication_index(), np.float32)
        return replica_ctx.all_reduce(reduce_op, x)

      return all_reduce_function

    report = tu.ReportJSON(self, eager_mode=True, replicated=True, use_hw=True)
    report.reset()

    with strategy.scope():
      summed = strategy.experimental_run_v2(
          make_all_reduce_function(reduce_util.ReduceOp.SUM))
      self.assertEqual(1.0, summed.numpy())

      mean = strategy.experimental_run_v2(
          make_all_reduce_function(reduce_util.ReduceOp.MEAN))
      self.assertEqual(0.5, mean.numpy())

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.run_v2_only
  def test_optimizer(self):
    strategy = ipu_strategy.IPUStrategy()

    report = tu.ReportJSON(self, eager_mode=True, replicated=True, use_hw=True)
    report.reset()

    with strategy.scope():
      initial_variable = 2.0
      variable = variables.Variable(initial_variable)
      learning_rate = 0.5
      num_iterations = 3

      data = [1.0, 2.0]
      dataset = dataset_ops.Dataset.from_tensor_slices((data))
      dataset = dataset.repeat(num_iterations)
      infeed = ipu_infeed_queue.IPUInfeedQueue(dataset,
                                               feed_name="feed",
                                               replication_factor=2)

      optimizer = keras.optimizer_v2.gradient_descent.SGD(learning_rate)

      @def_function.function(experimental_compile=True)
      def apply_gradient():
        gradient = infeed._dequeue()  # pylint: disable=protected-access
        optimizer.apply_gradients([(gradient, variable)])

      # The optimizers in v2 will sum the gradients, and not average them.
      expected_gradient = np.sum(data)
      expected_variable = initial_variable

      infeed.initializer  # pylint: disable=pointless-statement

      for _ in range(num_iterations):
        strategy.experimental_run_v2(apply_gradient)
        expected_variable -= learning_rate * expected_gradient
        self.assertEqual(expected_variable, variable.numpy())


if __name__ == "__main__":
  test.main()
