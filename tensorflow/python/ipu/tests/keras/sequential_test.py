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

from tensorflow import GradientTape
from tensorflow import keras
from tensorflow import random
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.eager import context
from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops

from tensorflow.keras import layers
from tensorflow.python import ipu


class KerasSequentialTest(test_util.TensorFlowTestCase):
  def tearDown(self):
    super().tearDown()
    # Reset the eager context to avoid polluting future test cases.
    context._reset_context()  # pylint: disable=protected-access

  @test_util.run_v2_only
  def testAddKnownShapeInScope(self):
    """The whole model creation is placed inside an IPU device scope,
    therefore all the layers are placed on the IPU.
    """
    with ops.device("/device:IPU:0"):
      report = tu.ReportJSON(self, eager_mode=True)
      report.reset()

      model = keras.Sequential()
      input_shape = [16, 16]
      model.add(layers.Masking(mask_value=0., input_shape=input_shape))
      model.add(layers.Conv1D(8, kernel_size=3, activation="relu"))

      report.assert_compiled_for_ipu("add() should have compiled some IPU ops")
      self.assertTrue(model.weights, "Weights should have been initialized")

      model.build(input_shape=None)
      report.assert_no_event(
          "Building a graph with a known input_shape should be a no-op")

  @test_util.run_v2_only
  def testAddUnknownShapeInScope(self):
    """The whole model creation is placed inside an IPU device scope,
    therefore all the layers are placed on the IPU.
    However the input_shape is only provided at build time.
    """
    with ops.device("/device:IPU:0"):
      report = tu.ReportJSON(self, eager_mode=True)
      report.reset()

      model = keras.Sequential()
      input_shape = [16, 16]
      different_input_shape = [8, 8]
      batch_size = 1
      different_batch_size = 2
      model.add(layers.Masking(mask_value=0.))
      model.add(layers.Conv1D(8, kernel_size=3, activation="relu"))
      report.assert_no_event(
          "Input size unknown: add() shouldn't have compiled any IPU op")

      with self.assertRaisesRegex(ValueError, "have not yet been created"):
        len(model.weights)
      model.build(input_shape=[batch_size] + input_shape)

      report.assert_compiled_for_ipu(
          "build() should have compiled some IPU ops")
      self.assertTrue(model.weights, "Weights should have been initialized")

      model.build(input_shape=[different_batch_size] + input_shape)
      report.assert_no_event(
          "Change in batch size should not trigger a rebuild")

      with self.assertRaisesRegex(ValueError,
                                  "is incompatible with the layer"):
        model.build(input_shape=[batch_size] + different_input_shape)

  @test_util.run_v2_only
  def testExecuteUnknownShapeInScope(self):
    """The whole model creation is placed inside an IPU device scope,
    therefore all the layers are placed on the IPU.
    However the input_shape is only provided at run time.
    """
    with ops.device("/device:IPU:0"):
      report = tu.ReportJSON(self, eager_mode=True)
      report.reset()

      model = keras.Sequential()
      input_shape = [16, 16]
      different_input_shape = [8, 8]
      batch_size = 1
      different_batch_size = 2
      model.add(layers.Masking(mask_value=0.))
      model.add(layers.Conv1D(8, kernel_size=3, activation="relu"))
      report.assert_no_event(
          "Input size unknown: add() shouldn't have compiled any IPU op")

      with self.assertRaisesRegex(ValueError, "have not yet been created"):
        len(model.weights)

      x = random.uniform([batch_size] + input_shape)
      model(x)

      self.assertTrue(model.weights, "Weights should have been initialized")
      report.assert_compiled_for_ipu(
          "call() should have compiled some IPU ops")

      model.build(input_shape=[different_batch_size] + input_shape)
      report.assert_no_event(
          "Change in batch size should not trigger a rebuild")

      with self.assertRaisesRegex(ValueError,
                                  "is incompatible with the layer"):
        model.build(input_shape=[batch_size] + different_input_shape)

  @test_util.run_v2_only
  def testAddKnownShapeIpuSequential(self):
    """ipu.Sequential is used therefore all the layers are placed on the IPU.
    """
    report = tu.ReportJSON(self, eager_mode=True)
    report.reset()

    model = ipu.Sequential()
    input_shape = [16, 16]
    model.add(layers.Masking(mask_value=0., input_shape=input_shape))
    model.add(layers.Conv1D(8, kernel_size=3, activation="relu"))

    report.assert_compiled_for_ipu("add() should have compiled some IPU ops")
    self.assertTrue(model.weights, "Weights should have been initialized")

    model.build(input_shape=None)
    report.assert_no_event(
        "Building a graph with a known input_shape should be a no-op")

  @test_util.run_v2_only
  def testAddUnknownShapeIpuSequential(self):
    """ipu.Sequential is used therefore all the layers are placed on the IPU.
    However the input_shape is only provided at build time.
    """
    report = tu.ReportJSON(self, eager_mode=True)
    report.reset()

    model = ipu.Sequential()
    input_shape = [16, 16]
    different_input_shape = [8, 8]
    batch_size = 1
    different_batch_size = 2
    model.add(layers.Masking(mask_value=0.))
    model.add(layers.Conv1D(8, kernel_size=3, activation="relu"))
    report.assert_no_event(
        "Input size unknown: add() shouldn't have compiled any IPU op")

    with self.assertRaisesRegex(ValueError, "have not yet been created"):
      len(model.weights)
    model.build(input_shape=[batch_size] + input_shape)

    report.assert_compiled_for_ipu("build() should have compiled some IPU ops")
    self.assertTrue(model.weights, "Weights should have been initialized")

    model.build(input_shape=[different_batch_size] + input_shape)
    report.assert_no_event("Change in batch size should not trigger a rebuild")

    with self.assertRaisesRegex(ValueError, "is incompatible with the layer"):
      model.build(input_shape=[batch_size] + different_input_shape)

  @test_util.run_v2_only
  def testExecuteUnknownShapeIpuSequential(self):
    """ipu.Sequential is used therefore all the layers are placed on the IPU.
    However the input_shape is only provided at run time.
    """
    report = tu.ReportJSON(self, eager_mode=True)
    report.reset()

    model = ipu.Sequential()
    input_shape = [16, 16]
    different_input_shape = [8, 8]
    batch_size = 1
    different_batch_size = 2
    model.add(layers.Masking(mask_value=0.))
    model.add(layers.Conv1D(8, kernel_size=3, activation="relu"))
    report.assert_no_event(
        "Input size unknown: add() shouldn't have compiled any IPU op")

    with self.assertRaisesRegex(ValueError, "have not yet been created"):
      len(model.weights)

    x = random.uniform([batch_size] + input_shape)
    model(x)

    self.assertTrue(model.weights, "Weights should have been initialized")
    report.assert_compiled_for_ipu("call() should have compiled some IPU ops")

    model.build(input_shape=[different_batch_size] + input_shape)
    report.assert_no_event("Change in batch size should not trigger a rebuild")

    with self.assertRaisesRegex(ValueError, "is incompatible with the layer"):
      model.build(input_shape=[batch_size] + different_input_shape)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
