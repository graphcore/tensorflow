# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow import keras
from tensorflow.compat.v1 import disable_v2_behavior
from tensorflow.python import ipu
from tensorflow.python.client import session
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.training import gradient_descent

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu

from test_optimizer import KernelLoggingOptimizer, AssertAllWeightsReplicaIdentical

disable_v2_behavior()


def make_mnist_dataset():
  def normalize(x, y):
    return x.astype("float16") / 255.0, y.astype("int32")

  def generator():
    return zip(x_train, y_train)

  mnist = keras.datasets.mnist
  train_data, _ = mnist.load_data()
  x_train, y_train = normalize(*train_data)
  types = x_train.dtype, y_train.dtype
  shapes = x_train.shape[1:], y_train.shape[1:]

  n_examples = len(x_train)
  dataset = dataset_ops.Dataset.from_generator(generator, types, shapes)
  dataset = dataset.shuffle(n_examples)
  dataset = dataset.batch(32, drop_remainder=True)
  dataset = dataset.repeat()
  return dataset


# This test is intended to verify that we get the same weight values produced on each replica
# when running simple models with the experimental.enable_prng_stability flag enabled.
@test_util.deprecated_graph_mode_only
class PrngStabilityModelsTest(test_util.TensorFlowTestCase):
  def setUp(self):
    # Reset to avoid global variables being used across tests.
    ops.reset_default_graph()

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 4
    cfg.floating_point_behaviour.esr = True
    cfg.experimental.enable_prng_stability = True
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    self.outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

  def createLoggingOptimizer(self, wrapped_optimizer):
    return KernelLoggingOptimizer(self.outfeed, wrapped_optimizer)

  def assertAllWeightsReplicaIdentical(self, out_queue, expected_replicas=4):
    self.assertTrue(out_queue, "Expected some logged variables.")
    for var in out_queue:
      AssertAllWeightsReplicaIdentical(self, var, expected_replicas)

  @tu.test_uses_ipus(num_ipus=4)
  def testSimpleLinearModel(self):
    with ipu.scopes.ipu_scope("/device:IPU:0"):

      def training_loop():
        def model(loss):
          x = constant_op.constant([[1], [2], [3], [4]], dtype=np.float16)
          y_true = constant_op.constant([[0], [-1], [-2], [-3]],
                                        dtype=np.float16)

          linear_model = keras.layers.Dense(units=1)

          y_pred = linear_model(x)
          loss = losses.mean_squared_error(labels=y_true, predictions=y_pred)

          # Doesn't need CrossReplicaOptimizer since the data is indentical on each replica.
          optimizer = self.createLoggingOptimizer(
              gradient_descent.GradientDescentOptimizer(0.01))
          train = optimizer.minimize(loss)
          with ops.control_dependencies([train]):
            return array_ops.identity(loss)

        loss = 0.0
        return ipu.loops.repeat(100, model, (loss))

      compiled = ipu.ipu_compiler.compile(training_loop)

    init = variables.global_variables_initializer()
    dequeue = self.outfeed.dequeue()

    with session.Session() as sess:
      sess.run(init)
      sess.run(compiled)

      out_queue = sess.run(dequeue)
      self.assertAllWeightsReplicaIdentical(out_queue)

  @tu.test_uses_ipus(num_ipus=4)
  def testSimpleDataDependency(self):
    with ipu.scopes.ipu_scope("/device:IPU:0"):

      def training_loop():
        def model(loss):
          rep_id = ipu.replication_ops.replication_index()
          x = control_flow_ops.cond(
              math_ops.equal(rep_id, 0), lambda: constant_op.constant(
                  [[1], [1], [1], [1]], dtype=np.float16),
              lambda: constant_op.constant([[1], [2], [3], [4]],
                                           dtype=np.float16))
          y_true = constant_op.constant([[0], [-1], [-2], [-3]],
                                        dtype=np.float16)

          linear_model = keras.layers.Dense(units=1)
          y_pred = linear_model(x)
          loss = losses.mean_squared_error(labels=y_true, predictions=y_pred)

          # Need to use CrossReplicaOptimizer since the conditional will cause the replicas to have different data.
          optimizer = self.createLoggingOptimizer(
              ipu.optimizers.CrossReplicaOptimizer(
                  gradient_descent.GradientDescentOptimizer(0.01)))
          train = optimizer.minimize(loss)
          with ops.control_dependencies([train]):
            return array_ops.identity(loss)

        loss = 0.0
        return ipu.loops.repeat(100, model, (loss))

      compiled = ipu.ipu_compiler.compile(training_loop)

    init = variables.global_variables_initializer()
    dequeue = self.outfeed.dequeue()

    with session.Session() as sess:
      sess.run(init)
      sess.run(compiled)

      out_queue = sess.run(dequeue)
      self.assertAllWeightsReplicaIdentical(out_queue)

  @tu.test_uses_ipus(num_ipus=4)
  def testFCModel(self):
    dataset = make_mnist_dataset()
    infeed = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    with ipu.scopes.ipu_scope("/device:IPU:0"):

      def training_loop():
        def model(loss, images, labels):
          x = keras.layers.Flatten()(images)
          x = keras.layers.Dense(256, activation=nn.relu, name="dense1")(x)
          x = keras.layers.Dense(128, activation=nn.relu, name="dense2")(x)
          logits = keras.layers.Dense(10, activation=nn.relu, name="dense3")(x)
          cross_entropy = nn.sparse_softmax_cross_entropy_with_logits(
              labels=labels, logits=logits)
          loss = math_ops.reduce_mean(cross_entropy)

          optimizer = self.createLoggingOptimizer(
              ipu.optimizers.CrossReplicaOptimizer(
                  gradient_descent.GradientDescentOptimizer(0.01)))
          train = optimizer.minimize(loss)
          with ops.control_dependencies([train]):
            return array_ops.identity(loss)

        loss = np.float16(0.0)
        return ipu.loops.repeat(100, model, (loss), infeed_queue=infeed)

      compiled = ipu.ipu_compiler.compile(training_loop)

    init = variables.global_variables_initializer()
    dequeue = self.outfeed.dequeue()

    with session.Session() as sess:
      sess.run(init)
      sess.run(infeed.initializer)
      sess.run(compiled)

      out_queue = sess.run(dequeue)
      self.assertAllWeightsReplicaIdentical(out_queue)

  @tu.test_uses_ipus(num_ipus=4)
  def testPipelinedFCModel(self):
    dataset = make_mnist_dataset()
    infeed = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    loss_out = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    with ipu.scopes.ipu_scope("/device:IPU:0"):

      def model():
        def stage1(images, labels):
          x = keras.layers.Flatten()(images)
          x = keras.layers.Dense(256, activation=nn.relu, name="dense1")(x)
          x = keras.layers.Dense(128, activation=nn.relu, name="dense2")(x)
          return x, labels

        def stage2(inputs, labels):
          logits = keras.layers.Dense(10, activation=nn.relu,
                                      name="dense3")(inputs)
          cross_entropy = nn.sparse_softmax_cross_entropy_with_logits(
              labels=labels, logits=logits)
          loss = math_ops.reduce_mean(cross_entropy)
          return loss

        def optimizer_func(loss):
          optimizer = self.createLoggingOptimizer(
              ipu.optimizers.CrossReplicaOptimizer(
                  gradient_descent.GradientDescentOptimizer(0.01)))
          return ipu.pipelining_ops.OptimizerFunctionOutput(optimizer, loss)

        return ipu.pipelining_ops.pipeline(
            computational_stages=[stage1, stage2],
            gradient_accumulation_count=16,
            repeat_count=3,
            infeed_queue=infeed,
            outfeed_queue=loss_out,
            optimizer_function=optimizer_func,
            outfeed_loss=True)

      compiled = ipu.ipu_compiler.compile(model)

    init = variables.global_variables_initializer()
    dequeue = self.outfeed.dequeue()

    with session.Session() as sess:
      sess.run(init)
      sess.run(infeed.initializer)
      sess.run(compiled)

      out_queue = sess.run(dequeue)
      self.assertAllWeightsReplicaIdentical(out_queue, expected_replicas=2)


if __name__ == "__main__":
  googletest.main()
