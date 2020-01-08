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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow import GradientTape
from tensorflow import keras
from tensorflow import random
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops

from tensorflow.keras import layers
from tensorflow.python import ipu


class KerasSequentialTest(test_util.TensorFlowTestCase):
  @test_util.run_v2_only
  def testAddKnownShapeInScope(self):
    """The whole model creation is placed inside an IPU device scope,
    therefore all the layers are placed on the IPU.
    """
    with ops.device("/device:IPU:0"):
      report = tu.ReportJSON(self, eager_mode=True)
      model = keras.Sequential()
      model.add(layers.Dense(64, input_shape=(3, 3), activation='relu'))
      model.add(layers.Dense(64, activation='relu'))
      model.add(layers.Dense(10, activation='softmax'))
      compute_loss = keras.losses.SparseCategoricalCrossentropy(
          from_logits=True)
      optimizer = keras.optimizers.SGD()

      def fn(x, y):
        with GradientTape() as tape:
          logits = model(x)
          loss = compute_loss(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

      x = random.uniform((3, 3))
      y = [0, 0, 1]

      report.reset()
      fn(x, y)
      types, _ = report.get_ipu_events()
      self.assertTrue(types, "fn() did not run on the IPU")

  @test_util.run_v2_only
  def testAddKnownShape(self):
    """The input shape of this sequential network is known therefore layers
    are instantiated inside the `add()` method. As a result we need to use our
    own type of Sequential model to make sure the layers are placed on the IPU.
    """
    report = tu.ReportJSON(self, eager_mode=True)

    model = ipu.Sequential()
    model.add(layers.Dense(64, input_shape=(3, 3), activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    compute_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.SGD()

    def fn(x, y):
      with GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(y, logits)
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      return loss

    x = random.uniform((3, 3))
    y = [0, 0, 1]

    report.reset()
    fn(x, y)
    types, _ = report.get_ipu_events()
    self.assertTrue(types, "fn() did not run on the IPU")

  @test_util.run_v2_only
  def testAddUnknownShape(self):
    """ The input shape in this test is unknown therefore the layers are
    actually instantiated inside the fn() function when the model is being
    executed. At which point the @ipu.function decorator will make sure the
    layers are placed on the ipu.
    """
    report = tu.ReportJSON(self, eager_mode=True)

    model = keras.Sequential()
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    compute_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.SGD()

    @ipu.function
    def fn(x, y):
      with GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(y, logits)
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      return loss

    x = random.uniform((3, 3))
    y = [0, 0, 1]

    report.reset()
    fn(x, y)
    types, _ = report.get_ipu_events()
    self.assertTrue(types, "fn() did not run on the IPU")


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
