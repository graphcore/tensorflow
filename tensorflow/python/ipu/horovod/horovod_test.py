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

import os
import numpy as np
from tensorflow.python import keras
from tensorflow.python import ops
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute.reduce_util import ReduceOp
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.horovod.ipu_horovod_strategy import IPUHorovodStrategy
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer


class HorovodTest(test_util.TensorFlowTestCase):
  @classmethod
  def setUpClass(cls):
    hvd.init()

  @classmethod
  def tearDownClass(cls):
    hvd.shutdown()

  def assertAllRanksEqual(self, local_value, name=None):
    """Assert that the current rank has the same value as the root rank."""
    with ops.Graph().as_default(), session.Session():
      local_tensor = constant_op.constant(local_value)
      root_tensor = hvd.broadcast(local_tensor, root_rank=0)
      root_value = root_tensor.eval()
      np.testing.assert_equal(local_value, root_value, name)

  @test_util.deprecated_graph_mode_only
  def test_basics(self):
    self.assertTrue(hvd.mpi_built())
    self.assertTrue(hvd.mpi_enabled())

    self.assertFalse(hvd.nccl_built())
    self.assertFalse(hvd.ddl_built())
    self.assertFalse(hvd.gloo_built())
    self.assertFalse(hvd.gloo_enabled())

    self.assertEqual(hvd.rank(), int(os.environ["OMPI_COMM_WORLD_RANK"]))

    self.assertEqual(hvd.local_rank(),
                     int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]))

    self.assertEqual(hvd.size(), hvd.local_size())
    self.assertTrue(hvd.is_homogeneous())

  @test_util.deprecated_graph_mode_only
  def test_collectives(self):
    rank = constant_op.constant(hvd.rank(), dtype=np.float32)
    summed = hvd.allreduce(rank, op=hvd.Sum)
    averaged = hvd.allreduce(rank)
    allgathered = hvd.allgather(array_ops.expand_dims(rank, axis=0))
    broadcast = hvd.broadcast(rank, root_rank=0)

    with self.assertRaisesRegex(NotImplementedError,
                                "The Adasum reduction is not implemented"):
      hvd.allreduce(rank, op=hvd.Adasum)

    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    with session.Session() as sess:
      self.assertAllEqual(np.arange(hvd.size()), sess.run(allgathered))
      self.assertAllEqual(np.sum(np.arange(hvd.size())), sess.run(summed))
      self.assertAllEqual(np.mean(np.arange(hvd.size())), sess.run(averaged))
      self.assertAllEqual(0.0, sess.run(broadcast))

  @test_util.deprecated_graph_mode_only
  def test_ipu_horovod_strategy(self):
    hvd_size = hvd.size()
    hvd_rank = hvd.rank()

    strategy = IPUHorovodStrategy()
    self.assertEqual(strategy.num_replicas_in_sync, hvd_size)

    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    with strategy.scope():

      def per_replica_fn():
        w = variable_scope.get_variable(name="w", initializer=hvd_rank + 1.0)
        self.assertEqual("/replica:0/task:0/device:IPU:0", w.device)
        return w * w

      per_replica_val = strategy.experimental_run_v2(per_replica_fn)
      strategy_sum = strategy.reduce(ReduceOp.SUM, per_replica_val)
      strategy_mean = strategy.reduce(ReduceOp.MEAN, per_replica_val)

      with session.Session() as sess:
        sess.run(variables.global_variables_initializer())

        # All workers should have the initial value from the first worker.
        self.assertEqual([1.0], sess.run(variables.global_variables()))
        self.assertEqual(1.0 * hvd_size, strategy_sum.eval())
        self.assertEqual(1.0, strategy_mean.eval())

  @test_util.deprecated_graph_mode_only
  def test_pipelining(self):
    gradient_accumulation_count = 4
    local_batch_size = 2

    features = np.ones((1, 20), dtype=np.float32) * hvd.rank()
    labels = np.ones(1, dtype=np.int32) * hvd.rank()
    dataset = dataset_ops.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.repeat().batch(local_batch_size, drop_remainder=True)

    loss_vals = []

    strategy = IPUHorovodStrategy()

    with strategy.scope():

      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def stage1(lr, images, labels):
        partial = keras.layers.Dense(32, activation="relu")(images)
        partial = keras.layers.Dense(16, activation="relu")(partial)
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
            device_mapping=[0, 0],
            gradient_accumulation_count=gradient_accumulation_count,
            inputs=[lr],
            infeed_queue=infeed_queue,
            repeat_count=2,
            outfeed_queue=outfeed_queue,
            optimizer_function=optimizer_function,
            name="Pipeline")
        return pipeline_op

      def compiled_model(lr):
        with ipu_scope("/device:IPU:0"):
          return ipu_compiler.compile(model, inputs=[lr])

      with ops.device("cpu"):
        lr = array_ops.placeholder(np.float32, [])

      train_op = strategy.experimental_run_v2(compiled_model, args=[lr])

      _, per_worker_losses = outfeed_queue.dequeue()

      # Mean across the local `gradient_accumulation_count` batches:
      per_worker_loss = math_ops.reduce_mean(per_worker_losses)

      # Global mean across the distributed workers (since it is already
      # divided by the global batch size above, we do a sum here):
      global_loss = strategy.reduce(ReduceOp.SUM, per_worker_loss)

      config = IPUConfig()
      config.auto_select_ipus = 1
      config.configure_ipu_system()
      ipu_utils.move_variable_initialization_to_cpu()

      with session.Session() as sess:
        sess.run(infeed_queue.initializer)
        sess.run(variables.global_variables_initializer())

        for _ in range(10):
          sess.run(train_op, {lr: 0.01})
          global_loss_val = sess.run(global_loss)

          if loss_vals:
            # Check that the loss decreases monotonically.
            self.assertLess(global_loss_val, loss_vals[-1])
          loss_vals.append(global_loss_val)

        sess.run(infeed_queue.deleter)
        sess.run(outfeed_queue.deleter)

        # Check all variables are equal across workers.
        for variable in variables.global_variables():
          self.assertAllRanksEqual(variable.eval(), variable.name)


if __name__ == "__main__":
  test.main()
