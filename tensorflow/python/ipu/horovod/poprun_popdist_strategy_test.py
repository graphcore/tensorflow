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

from absl.testing import parameterized

import popdist
import popdist.tensorflow

import tensorflow as tf
from tensorflow.python.ipu import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ipu.horovod import popdist_strategy
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras import layers
from tensorflow.python.platform import test


def test_dataset(length=None, batch_size=1, x_val=1.0, y_val=0.2):
  constant_d = constant_op.constant(x_val, shape=[32])
  constant_l = constant_op.constant(y_val, shape=[1])

  ds = dataset_ops.Dataset.from_tensors((constant_d, constant_l))
  ds = ds.repeat(length)
  ds = ds.shard(num_shards=popdist.getNumInstances(),
                index=popdist.getInstanceIndex())
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


def simple_model():
  inputs = Input(shape=(32,))
  outputs = layers.Dense(1)(inputs)

  return Model(inputs, outputs)


class PoprunPopDistStrategyTest(test_util.TensorFlowTestCase,
                                parameterized.TestCase):
  TESTCASES = [{
      "testcase_name": "with_asynchronous_callbacks",
      "enable_asynchronous_callbacks": True,
  }, {
      "testcase_name": "without_asynchronous_callbacks",
      "enable_asynchronous_callbacks": False,
  }]

  @parameterized.named_parameters(*TESTCASES)
  @tu.test_uses_ipus(num_ipus=2)
  def test_single_multi_replica_training_step_keras(
      self, enable_asynchronous_callbacks):
    config = ipu.config.IPUConfig()
    popdist.tensorflow.set_ipu_config(config, ipus_per_replica=1)
    config.configure_ipu_system()

    hvd.init()

    strategy = popdist_strategy.PopDistStrategy()

    with strategy.scope():
      learning_rate = 0.5
      model = simple_model()
      model.set_asynchronous_callbacks(enable_asynchronous_callbacks)
      optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

      @tf.function(jit_compile=True)
      def loss_fn(_, y_pred):
        return y_pred

      num_replicas = popdist.getNumTotalReplicas()
      model.compile(loss=loss_fn,
                    optimizer=optimizer,
                    steps_per_execution=num_replicas)
      model.build((1, 32))

      model.fit(test_dataset(20), epochs=1, verbose=False)

      model.evaluate(test_dataset(20), verbose=False)

      model.predict(test_dataset(20, batch_size=4), verbose=False)


if __name__ == "__main__":
  test.main()
