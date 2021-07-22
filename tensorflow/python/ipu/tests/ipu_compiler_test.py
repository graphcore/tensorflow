# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_estimator
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import googletest


class IpuCompilerTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testCompileWrongDeviceRaisesException(self):
    def my_net(a):
      return a + a

    with ops.device("/device:CPU:0"):
      a = array_ops.placeholder(np.float32, shape=[1])
      with self.assertRaisesRegex(Exception, "not placed on an IPU"):
        ipu_compiler.compile(my_net, inputs=[a])

  @test_util.deprecated_graph_mode_only
  def testCompileNoopOnWrongDeviceIsOK(self):
    def my_net():
      return control_flow_ops.no_op()

    with ops.device("/device:CPU:0"):
      ipu_compiler.compile(my_net)

  @test_util.deprecated_graph_mode_only
  def testCompileForDevicesInWorkerTask(self):
    def my_net(a):
      return a + a

    with ops.device("/job:worker/task:0"):
      with ops.device("/device:IPU:0"):
        a = array_ops.placeholder(np.float32, shape=[1])
        ipu_compiler.compile(my_net, inputs=[a])

      with ops.device("/device:CPU:0"):
        a = array_ops.placeholder(np.float32, shape=[1])
        with self.assertRaisesRegex(Exception, "not placed on an IPU"):
          ipu_compiler.compile(my_net, inputs=[a])

  @test_util.deprecated_graph_mode_only
  def testCompileWrongUseOfEstimatorSpec(self):
    def my_net(features, labels):
      mode = ModeKeys.TRAIN
      loss = features + labels
      train_op = array_ops.identity(loss)
      return [
          2, [], 1,
          [model_fn.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)]
      ]

    a = array_ops.placeholder(np.float32, shape=[1])
    b = array_ops.placeholder(np.float32, shape=[1])
    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegex(ValueError, "contains an EstimatorSpec"):
        ipu_compiler.compile(my_net, inputs=[a, b])

  @test_util.deprecated_graph_mode_only
  def testCompileWrongUseOfIPUEstimatorSpec(self):
    def my_net(features, labels):
      mode = ModeKeys.TRAIN
      loss = features + labels
      train_op = array_ops.identity(loss)
      return [
          2, [], 1,
          [
              ipu_estimator.IPUEstimatorSpec(mode=mode,
                                             loss=loss,
                                             train_op=train_op)
          ]
      ]

    a = array_ops.placeholder(np.float32, shape=[1])
    b = array_ops.placeholder(np.float32, shape=[1])
    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegex(ValueError, "contains an EstimatorSpec"):
        ipu_compiler.compile(my_net, inputs=[a, b])


if __name__ == "__main__":
  googletest.main()
