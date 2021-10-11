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
# ==============================================================================
import numpy as np

import popdist
import popdist.tensorflow

import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.framework import constant_op, test_util
from tensorflow.python.ipu.horovod import ipu_multi_replica_strategy
from tensorflow.python.platform import test
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu import ipu_strategy


class DistributedTF2Test(test_util.TensorFlowTestCase):
  def assert_all_instances_equal(self, local_value, name=None):
    # Assert that the current instance has the same value as the root instance.
    local_tensor = constant_op.constant(local_value)
    root_tensor = hvd.broadcast(local_tensor, root_rank=0)
    np.testing.assert_equal(local_value, root_tensor.numpy(), name)

  def assert_all_instances_not_equal(self, local_value):
    local_tensor = constant_op.constant(local_value)
    root_tensor = hvd.broadcast(local_tensor, root_rank=0)

    if hvd.local_rank() == 0:
      return

    assert not np.equal(local_value, root_tensor.numpy()).any()

  def prepare_model(self):
    # Make sure we have different parameters on each index
    bias = tf.keras.initializers.Constant(value=popdist.getInstanceIndex())

    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(4,
                               3,
                               activation='relu',
                               bias_initializer=bias,
                               name='test_bias'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2),
    ])

  def prepare_dataset(self):
    def generator():
      for _ in range(100):
        yield np.random.rand(4, 4, 1), np.random.randint(1, 2, size=1)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=((4, 4, 1), (1,)),
    )

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
      tf.data.experimental.AutoShardPolicy.OFF
    dataset = dataset.with_options(options)

    dataset = dataset.shard(num_shards=popdist.getNumInstances(),
                            index=popdist.getInstanceIndex())
    dataset = dataset.batch(10, drop_remainder=True)

    return dataset

  def test_tf2_distributed_ipu_strategy(self):
    config = ipu.config.IPUConfig()
    popdist.tensorflow.set_ipu_config(config, ipus_per_replica=1)
    config.configure_ipu_system()

    hvd.init()

    strategy = ipu_strategy.IPUStrategy()

    with strategy.scope():
      dataset = self.prepare_dataset()
      model = self.prepare_model()
      optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
      loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

      model.compile(optimizer=optimizer,
                    loss=loss_fn,
                    steps_per_execution=popdist.getNumTotalReplicas())

      # Build the model separately so we can assert that the biases are
      # broadcasted properly before training.
      model.build((1, 4, 4, 1))

      layer = model.get_layer(name='test_bias')
      self.assert_all_instances_not_equal(layer.get_weights()[1])

      history = model.fit(dataset,
                          steps_per_epoch=popdist.getNumTotalReplicas(),
                          epochs=1)

      # Make sure the losses and weights are not equal
      self.assert_all_instances_not_equal(history.history['loss'])
      self.assert_all_instances_not_equal(layer.get_weights()[1])

  def test_tf2_distributed_ipu_multi_replica_strategy(self):
    config = ipu.config.IPUConfig()
    popdist.tensorflow.set_ipu_config(config, ipus_per_replica=1)
    config.configure_ipu_system()

    hvd.init()

    strategy = ipu_multi_replica_strategy.IPUMultiReplicaStrategy()

    with strategy.scope():
      dataset = self.prepare_dataset()
      model = self.prepare_model()
      optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
      loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

      model.compile(optimizer=optimizer,
                    loss=loss_fn,
                    steps_per_execution=popdist.getNumTotalReplicas())

      # Build the model separately so we can assert that the biases are
      # broadcasted properly before training.
      model.build((1, 4, 4, 1))

      layer = model.get_layer(name='test_bias')
      self.assert_all_instances_equal(layer.get_weights()[1])

      history = model.fit(dataset,
                          steps_per_epoch=popdist.getNumTotalReplicas(),
                          epochs=1)

      # Make sure the losses and weights are identical as we reduce over all
      # IPUs
      self.assert_all_instances_equal(history.history['loss'])

      for v in model.trainable_variables:
        self.assert_all_instances_equal(v)

  def test_single_multi_replica_training_step(self):
    config = ipu.config.IPUConfig()
    popdist.tensorflow.set_ipu_config(config, ipus_per_replica=1)
    config.configure_ipu_system()

    hvd.init()

    strategy = ipu_multi_replica_strategy.IPUMultiReplicaStrategy()

    with strategy.scope():
      learning_rate = 0.5
      initial_w = 2.0
      optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

      w = tf.Variable(initial_w)

      @tf.function(experimental_compile=True)
      def step_fn(x):
        with tf.GradientTape() as tape:
          loss = w * x
        optimizer.minimize(loss, var_list=[w], tape=tape)

        return loss

      @tf.function(experimental_compile=True)
      def step_fn_wrapper(x):
        per_replica_loss = strategy.run(step_fn, args=[x])
        loss_reduced = strategy.reduce(tf.distribute.ReduceOp.SUM,
                                       per_replica_loss)

        return loss_reduced

      num_replicas = popdist.getNumTotalReplicas()
      reference_w = initial_w

      for x in range(10):
        self.assertEqual(reference_w, w.numpy())
        with tf.device("/device:IPU:0"):
          loss_final = step_fn_wrapper(tf.constant(tf.cast(x, tf.float32)))
        self.assertEqual(num_replicas * reference_w * x, loss_final)

        # L(x) = num_replicas * w * x
        # dL(x)/dw = num_replicas * x
        # w := w - learning_rate * num_replicas * x
        reference_w -= learning_rate * num_replicas * x

  def test_single_multi_replica_training_step_keras(self):
    config = ipu.config.IPUConfig()
    popdist.tensorflow.set_ipu_config(config, ipus_per_replica=1)
    config.configure_ipu_system()

    hvd.init()

    strategy = ipu_multi_replica_strategy.IPUMultiReplicaStrategy()

    with strategy.scope():
      learning_rate = 0.5
      initial_w = 2.0
      model = tf.keras.Sequential([
          tf.keras.layers.Dense(
              1,
              kernel_initializer=tf.keras.initializers.Constant(initial_w),
              use_bias=False)
      ])
      optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

      @tf.function(experimental_compile=True)
      def loss_fn(_, y_pred):
        return y_pred

      num_replicas = popdist.getNumTotalReplicas()
      reference_w = initial_w
      model.compile(loss=loss_fn,
                    optimizer=optimizer,
                    steps_per_execution=num_replicas)
      model.build((1, 1))

      for x in range(10):
        self.assertEqual(reference_w,
                         model.trainable_variables[0][0][0].numpy())
        history = model.fit(
            np.array([[x]], np.float32).repeat(popdist.getNumLocalReplicas(),
                                               axis=0),
            np.array([[x]], np.float32).repeat(popdist.getNumLocalReplicas(),
                                               axis=0),
            steps_per_epoch=num_replicas,
            epochs=1)
        self.assertEqual(reference_w * x, history.history['loss'][0])

        # L(x) = w * x
        # dL(x)/dw = x
        # w := w - learning_rate * x
        reference_w -= learning_rate * x

  def test_single_training_step_equal_in_tf_and_keras(self):
    # This test verifies that a training loop in raw TensorFlow and Keras yield
    # the same losses, gradients and weight updates.

    def initialize_model_with_seed():
      # Make sure we initialize the kernels in a reproducible manner, create
      # an initializer with a constant seed.
      initializer = tf.keras.initializers.GlorotNormal(seed=1234)

      return tf.keras.models.Sequential([
          tf.keras.layers.Conv2D(4,
                                 3,
                                 kernel_initializer=initializer,
                                 use_bias=False,
                                 activation='relu'),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(2,
                                kernel_initializer=initializer,
                                use_bias=False),
      ])

    config = ipu.config.IPUConfig()
    popdist.tensorflow.set_ipu_config(config, ipus_per_replica=1)
    config.configure_ipu_system()

    hvd.init()

    strategy = ipu_multi_replica_strategy.IPUMultiReplicaStrategy()

    with strategy.scope():
      learning_rate = 0.01

      model_tf = initialize_model_with_seed()
      model_keras = initialize_model_with_seed()
      optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

      @tf.function(experimental_compile=True)
      def step_fn_tf(x, y):
        with tf.GradientTape() as tape:
          output = model_tf(x)
          loss = tf.keras.losses.sparse_categorical_crossentropy(
              y_true=y, y_pred=output, from_logits=True)
          loss = tf.nn.compute_average_loss(
              loss, global_batch_size=popdist.getNumTotalReplicas())
        optimizer.minimize(loss,
                           var_list=model_tf.trainable_variables,
                           tape=tape)

        return loss

      @tf.function(experimental_compile=True)
      def run_training_step_tf(x, y):
        per_replica_loss = strategy.run(step_fn_tf, args=[x, y])
        loss_reduced = strategy.reduce(tf.distribute.ReduceOp.SUM,
                                       per_replica_loss)

        return loss_reduced

      loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

      model_keras.compile(optimizer=optimizer,
                          loss=loss_fn,
                          steps_per_execution=popdist.getNumTotalReplicas())

      def run_training_step_keras(x, y):
        history = model_keras.fit(
            x, y, steps_per_epoch=popdist.getNumTotalReplicas(), epochs=1)

        return history.history['loss'][0]

      # First generate some random data using numpy, so we can reuse the same
      # data for both TF and Keras in order to force reproducibility.
      input_sample = np.random.uniform(0, 1, (1, 4, 4, 1))
      output_sample = np.random.randint(1, 2, size=1)

      # The Keras `.fit()` API requires the input to be replicated `num_replica`
      # times.
      x_tf = tf.constant(tf.cast(input_sample, tf.float32))
      y_tf = tf.constant(tf.cast(output_sample, tf.float32))
      x_keras = tf.constant(
          tf.cast(
              np.repeat(input_sample, popdist.getNumLocalReplicas(), axis=0),
              tf.float32))
      y_keras = tf.constant(
          tf.cast(
              np.repeat(output_sample, popdist.getNumLocalReplicas(), axis=0),
              tf.float32))

      # First test whether a single distributed training step yields the same
      # loss values for both TensorFlow and Keras.
      with tf.device("/device:IPU:0"):
        loss_final_tf = run_training_step_tf(x_tf, y_tf)
        loss_final_keras = run_training_step_keras(x_keras, y_keras)

      self.assertEqual(loss_final_tf, loss_final_keras)

      # Assert that both models have the same weights after the first backwards
      # pass.
      for i, _ in enumerate(model_tf.trainable_variables):
        np.testing.assert_equal(model_tf.trainable_variables[i].numpy(),
                                model_keras.trainable_variables[i].numpy())

      @tf.function(experimental_compile=True)
      def step_fn_eval_tf(x, y):
        output = model_tf(x, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=y, y_pred=output, from_logits=True)
        loss = tf.nn.compute_average_loss(
            loss, global_batch_size=popdist.getNumTotalReplicas())

        return loss

      @tf.function(experimental_compile=True)
      def run_eval_step_tf(x, y):
        per_replica_loss = strategy.run(step_fn_eval_tf, args=[x, y])
        loss_reduced = strategy.reduce(tf.distribute.ReduceOp.SUM,
                                       per_replica_loss)

        return loss_reduced

      def run_eval_step_keras(x, y):
        scores = model_keras.evaluate(x,
                                      y,
                                      batch_size=8,
                                      steps=popdist.getNumTotalReplicas())

        return scores

      # TODO(T47443) The `8 *` will be removed later, but it depends on a fix
      # for T47443.
      x_keras_eval = tf.constant(
          tf.cast(
              np.repeat(input_sample,
                        8 * popdist.getNumTotalReplicas(),
                        axis=0), tf.float32))
      y_keras_eval = tf.constant(
          tf.cast(
              np.repeat(output_sample,
                        8 * popdist.getNumTotalReplicas(),
                        axis=0), tf.float32))

      with tf.device("/device:IPU:0"):
        val_loss_final_tf = run_eval_step_tf(x_tf, y_tf)
        val_loss_final_keras = run_eval_step_keras(x_keras_eval, y_keras_eval)

      self.assertEqual(val_loss_final_tf, val_loss_final_keras)


if __name__ == "__main__":
  test.main()
