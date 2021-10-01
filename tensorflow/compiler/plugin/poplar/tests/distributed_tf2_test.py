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
    """Assert that the current instance has the same value as the root instance."""
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
        tf.keras.layers.Conv2D(32,
                               3,
                               activation='relu',
                               bias_initializer=bias,
                               name='test_bias'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10),
    ])

  def prepare_dataset(self):
    def generator():
      for _ in range(100):
        yield np.random.rand(32, 32, 1), np.random.randint(1, 10, size=1)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=((32, 32, 1), (1,)),
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

      # Build the model separately so we can assert that the biases are broadcasted properly before training.
      model.build((1, 32, 32, 1))

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

      # Build the model separately so we can assert that the biases are broadcasted properly before training.
      model.build((1, 32, 32, 1))

      layer = model.get_layer(name='test_bias')
      self.assert_all_instances_equal(layer.get_weights()[1])

      history = model.fit(dataset,
                          steps_per_epoch=popdist.getNumTotalReplicas(),
                          epochs=1)

      # Make sure the losses and weights are identical as we reduce over all IPUs
      self.assert_all_instances_equal(history.history['loss'])

      for v in model.trainable_variables:
        self.assert_all_instances_equal(v)


if __name__ == "__main__":
  test.main()
