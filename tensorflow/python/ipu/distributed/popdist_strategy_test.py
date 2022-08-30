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
import popdist
import numpy as np

from tensorflow import debugging
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute.reduce_util import ReduceOp
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu.distributed.popdist_strategy import PopDistStrategy, IPUSyncOnReadVariable, IPUMirroredVariable
from tensorflow.python.ops import variables
from tensorflow.python.ops.variable_scope import VariableAggregation, VariableSynchronization
from tensorflow.python.platform import test
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import math_ops


def simple_model():
  inputs = Input(shape=(32,))
  outputs = layers.Dense(1)(inputs)

  return Model(inputs, outputs)


def test_dataset(length=None, batch_size=1, x_val=1.0, y_val=0.2):
  constant_d = constant_op.constant(x_val, shape=[32])
  constant_l = constant_op.constant(y_val, shape=[1])

  ds = dataset_ops.Dataset.from_tensors((constant_d, constant_l))
  ds = ds.repeat(length)
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


class PopDistStrategyTest(test_util.TensorFlowTestCase):  # pylint: disable=abstract-method
  @classmethod
  def setUpClass(cls):
    hvd.init()
    # Instantiate here to call popdist.init() and popdist.finalizeBackend() once
    cls.strategy = PopDistStrategy()

  @classmethod
  def tearDownClass(cls):
    hvd.shutdown()

  def _create_test_objects(self, auto_select_ipus=1):
    config = IPUConfig()
    config.auto_select_ipus = auto_select_ipus
    config.configure_ipu_system()

    return config

  def test_update_ipu_config(self):
    config = self._create_test_objects()
    self.strategy.update_ipu_config(config)
    self.assertEqual(
        config.experimental.multi_replica_distribution.process_count,
        popdist.getNumInstances())
    self.assertEqual(
        config.experimental.multi_replica_distribution.process_index,
        popdist.getInstanceIndex())

  def test_strategy(self):
    config = self._create_test_objects()
    self.strategy.update_ipu_config(config)

    with self.strategy.scope():
      v = variables.Variable(initial_value=popdist.getInstanceIndex() + 1,
                             dtype=np.float32)
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

        return y_all_reduced

      per_replica_value = self.strategy.run(per_replica_fn,
                                            args=[constant_op.constant(2.0)])

      # This reduction is performed on CPU, and hence uses Horovod.
      value_all_reduced = self.strategy.reduce(ReduceOp.SUM, per_replica_value)

      # The initial value should be broadcast from rank 0.
      self.assertEqual(v, 1.0)

      # There should be one allreduce sum of the values.
      self.assertEqual(value_all_reduced, popdist.getNumInstances() * 2.0)

  def test_strategy_without_ipu_reduction(self):
    config = self._create_test_objects()
    self.strategy.update_ipu_config(config)

    # Modify to keep one instance of the strategy
    self.strategy._extended._add_ipu_cross_replica_reductions = False  # pylint: disable=protected-access

    with self.strategy.scope():
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
      self.strategy.run(per_replica_fn, args=[constant_op.constant(2.0)])

    # Set back to default value
    self.strategy._extended._add_ipu_cross_replica_reductions = True  # pylint: disable=protected-access

  def test_strategy_with_sync_on_read_variable(self):
    config = self._create_test_objects()
    self.strategy.update_ipu_config(config)

    with self.strategy.scope():
      w = variables.Variable(initial_value=float(popdist.getInstanceIndex() +
                                                 1),
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
      self.strategy.run(
          per_replica_fn,
          args=[constant_op.constant(popdist.getInstanceIndex() + 1.0)])
      debugging.assert_equal([2.5], w.read_value())

  def test_strategy_with_mirrored_variable(self):
    config = self._create_test_objects()
    self.strategy.update_ipu_config(config)

    with self.strategy.scope():
      w = variables.Variable(initial_value=float(popdist.getInstanceIndex() +
                                                 1),
                             dtype=np.float32,
                             synchronization=VariableSynchronization.ON_WRITE,
                             aggregation=VariableAggregation.SUM)

      @def_function.function
      def per_replica_fn():
        self.assertIsInstance(w, IPUMirroredVariable)
        return w * w

      per_replica_ret = self.strategy.run(per_replica_fn, args=[])
      sum_ret = self.strategy.reduce(ReduceOp.SUM, per_replica_ret, axis=None)
      self.assertEqual([1.0], per_replica_ret)
      self.assertEqual(2.0, sum_ret)

  def test_distribute_dataset(self):
    config = self._create_test_objects()
    self.strategy.update_ipu_config(config)

    dataset = dataset_ops.Dataset.range(10, output_type=np.float32)
    dataset = dataset.shard(num_shards=popdist.getNumInstances(),
                            index=popdist.getInstanceIndex())

    with self.strategy.scope():

      @def_function.function
      def step_fn(iterator):
        x = next(iterator)

        return x * x

      dist_iterator = ipu_infeed_queue.IPUIterator(dataset=dataset)

      def run_fn(iterator):
        per_replica_y = self.strategy.run(step_fn, args=[iterator])
        return self.strategy.reduce(ReduceOp.SUM, per_replica_y, axis=None)

      self.assertEqual(1.0, run_fn(dist_iterator))
      self.assertEqual(13.0, run_fn(dist_iterator))
      self.assertEqual(41.0, run_fn(dist_iterator))

  def test_all_reduce(self):
    config = self._create_test_objects()
    self.strategy.update_ipu_config(config)
    with self.strategy.scope():

      @def_function.function
      def per_replica_fn(x):
        return x * x

      per_replica_y = self.strategy.run(per_replica_fn,
                                        args=[popdist.getInstanceIndex() + 1])
      sum_y = self.strategy.reduce(ReduceOp.SUM, per_replica_y, axis=None)
      self.assertEqual(5, sum_y)  # 1*1 + 2*2

  def test_batch_normalization(self):
    config = self._create_test_objects()
    self.strategy.update_ipu_config(config)

    with self.strategy.scope():
      batch_norm = layers.BatchNormalization(momentum=0.0)

      @def_function.function
      def per_replica_fn(x):
        y = batch_norm(x, training=True)

        self.assertIsInstance(batch_norm.beta, IPUMirroredVariable)
        self.assertIsInstance(batch_norm.gamma, IPUMirroredVariable)

        self.assertIsInstance(batch_norm.moving_mean, IPUSyncOnReadVariable)
        self.assertIsInstance(batch_norm.moving_variance,
                              IPUSyncOnReadVariable)

        return y

      x = constant_op.constant([[2.0 * (popdist.getInstanceIndex() + 1)],
                                [0.0]])
      per_replica_y = self.strategy.run(per_replica_fn, args=(x,))
      sum_y = self.strategy.reduce(ReduceOp.SUM, per_replica_y, axis=None)

      # mean(mean(2, 0), mean(4, 0)) = mean(1, 3) = 1.5
      self.assertAllEqual([1.5], batch_norm.moving_mean)
      # mean(var(2, 0), var(4, 0)) = mean(1, 4) = 2.5
      self.assertAllEqual([2.5], batch_norm.moving_variance)

  def test_dataset_infeed(self):
    config = self._create_test_objects()
    self.strategy.update_ipu_config(config)

    dataset = dataset_ops.Dataset.from_tensor_slices([0.0]).repeat()
    # Test with a dataset host op.
    dataset = dataset.map(lambda x: x + popdist.getInstanceIndex())
    infeed_queue = iter(dataset)

    @def_function.function
    def net(iterator):
      s = 0.0
      for _ in math_ops.range(10):
        s += 2 * next(iterator)
      return s

    with self.strategy.scope():
      res = self.strategy.run(net, args=(infeed_queue,))
      self.assertEqual(popdist.getInstanceIndex() * 20, res)


if __name__ == "__main__":
  test.main()
