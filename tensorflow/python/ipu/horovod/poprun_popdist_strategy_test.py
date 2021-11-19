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
from tensorflow.python import ipu
from tensorflow.python.framework import test_util
from tensorflow.python.ipu.horovod import popdist_strategy
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.platform import test


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
  def test_single_multi_replica_training_step_keras(
      self, enable_asynchronous_callbacks):
    config = ipu.config.IPUConfig()
    popdist.tensorflow.set_ipu_config(config, ipus_per_replica=1)
    config.configure_ipu_system()

    hvd.init()

    strategy = popdist_strategy.PopDistStrategy()

    with strategy.scope():
      learning_rate = 0.5
      initial_w = 2.0
      model = tf.keras.Sequential([
          tf.keras.layers.Dense(
              8,
              kernel_initializer=tf.keras.initializers.Constant(initial_w),
              use_bias=False)
      ])
      model.set_asynchronous_callbacks(enable_asynchronous_callbacks)
      optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

      @tf.function(experimental_compile=True)
      def loss_fn(_, y_pred):
        return y_pred

      num_replicas = popdist.getNumTotalReplicas()
      model.compile(loss=loss_fn,
                    optimizer=optimizer,
                    steps_per_execution=num_replicas)
      model.build((1, 8))

      model.fit(np.array([[i for i in range(8)]],
                         np.float32).repeat(popdist.getNumLocalReplicas(),
                                            axis=0),
                np.array([[0]],
                         np.float32).repeat(popdist.getNumLocalReplicas(),
                                            axis=0),
                steps_per_epoch=num_replicas,
                epochs=1)

      model.evaluate(np.array([[i for i in range(8)]],
                              np.float32).repeat(popdist.getNumLocalReplicas(),
                                                 axis=0),
                     np.array([[0]], np.float32).repeat(
                         popdist.getNumLocalReplicas()),
                     steps=popdist.getNumTotalReplicas())

      model.predict(np.array([[i for i in range(8)]],
                             np.float32).repeat(popdist.getNumLocalReplicas(),
                                                axis=0),
                    steps=popdist.getNumTotalReplicas())


if __name__ == "__main__":
  test.main()
