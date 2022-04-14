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
from tensorflow.python.ipu.config import IPUConfig

from tensorflow.python.ipu import test_utils as tu
from tensorflow.python import keras
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import keras as ipu_keras


class PipelineModelReplicatedTest(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=4)
  @test_util.run_v2_only
  def testPredictWithNumpyDataBs2Replicas2(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 4
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()

    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32),
                                       dtype=np.single,
                                       batch_size=2)
      init = keras.initializers.Constant(0.1)
      with ipu_keras.PipelineStage(0):
        x = keras.layers.Dense(4, name="layer0",
                               kernel_initializer=init)(input_layer)
      with ipu_keras.PipelineStage(1):
        x = keras.layers.Dense(2, name="layer1", kernel_initializer=init)(x)
      m = keras.Model(input_layer, x)

      m.compile(steps_per_execution=12)

      # Input data
      input_x = np.full([96, 32], 1.0, dtype=np.single)

      # Generate predictions
      result = m.predict(input_x, batch_size=2)

      # The result is a Numpy array of predictions
      self.assertEqual(type(result), np.ndarray)
      self.assertEqual(result.shape, (96, 2))
      for i, r in enumerate(result):
        self.assertEqual(0, np.sum(r != result[i - 1]))


if __name__ == "__main__":
  googletest.main()
