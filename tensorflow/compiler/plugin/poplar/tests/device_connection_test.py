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
# ==============================================================================

import multiprocessing
import time
import numpy as np

from tensorflow.python.ipu.utils import DeviceConnectionType
from tensorflow.python.ops import variables
from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent

from tensorflow.python.ops import variable_scope


class Process:
  def __init__(self, fn, *args):
    self.q = multiprocessing.Queue()
    self.p = multiprocessing.Process(target=lambda: self.q.put(fn(*args)))
    self.p.start()

  def Join(self):
    self.p.join()
    return self.p.exitcode


def _ConfigureSystem(connection_type):
  cfg = ipu.config.IPUConfig()
  cfg.select_ipus = [0]  # Always use the same IPU.
  cfg.device_connection.version = 'ipu1'
  cfg.device_connection.type = connection_type
  cfg.configure_ipu_system()


ndims = 2
M = 3
N = 5
K = 7  # input features per group, output features per group, number of groups


def _MyNet():
  with variable_scope.variable_scope("vs", use_resource=True):
    with ops.device("cpu"):
      inp = array_ops.placeholder(np.float32, [1] + [24] * ndims + [M * K],
                                  name="input")
      bias = array_ops.placeholder(np.float32, [N * K], name="bias")
    with ops.device("/device:IPU:0"):
      weights = variable_scope.get_variable("weights",
                                            [8] * ndims + [M, N * K])
      output = nn.convolution(inp,
                              weights,
                              strides=[1] + [4] * ndims + [1],
                              padding="VALID",
                              name='cnv')
      output = nn.bias_add(output, bias, name='bias_add')
      loss = math_ops.reduce_sum(math_ops.square(output))
      optimizer = gradient_descent.GradientDescentOptimizer(0.0005)
      train = optimizer.minimize(loss)
      return train, loss, inp, bias


class TestDeviceConnection(xla_test.XLATestCase):  # pylint: disable=abstract-method
  @test_util.deprecated_graph_mode_only
  def testOfflineCompilation(self):
    if ipu.utils.running_on_ipu_model():
      self.skipTest(
          "There is no device contention with the model: nothing to test.")

    def BuildAndRunModelOffline():
      connection_type = DeviceConnectionType.NEVER
      with session.Session() as sess:
        train, loss, _, _ = _MyNet()
        _ConfigureSystem(connection_type)
        train = ipu.ipu_compiler.compile(lambda: (loss, train), [])

        with self.assertRaisesRegex(Exception,
                                    "configured for compilation only"):
          sess.run(variables.global_variables_initializer())

    p0 = Process(BuildAndRunModelOffline)
    p1 = Process(BuildAndRunModelOffline)
    exit_code0 = p0.Join()
    exit_code1 = p1.Join()
    self.assertEqual(exit_code0, 0)
    self.assertEqual(exit_code1, 0)

  @test_util.deprecated_graph_mode_only
  def testAlwaysCompilation(self):
    if ipu.utils.running_on_ipu_model():
      self.skipTest(
          "There is no device contention with the model: nothing to test.")

    def BuildAndRunModelAlways(first):
      connection_type = DeviceConnectionType.ALWAYS
      with session.Session() as sess:
        train, loss, inp, bias = _MyNet()
        if first:
          _ConfigureSystem(connection_type)
          train = ipu.ipu_compiler.compile(lambda: (loss, train), [])
          sess.run(variables.global_variables_initializer())
          fd = {
              inp: np.random.random_sample([1] + [24] * ndims + [M * K]),
              bias: np.random.random_sample([N * K])
          }
          sess.run(train, fd)
        else:
          time.sleep(1)  # Make sure the first process goes first.
          # IPU already in use by other process: configuration should fail.
          with self.assertRaisesRegex(Exception, "Could not attach"):
            _ConfigureSystem(connection_type)

    p0 = Process(BuildAndRunModelAlways, True)
    p1 = Process(BuildAndRunModelAlways, False)
    exit_code0 = p0.Join()
    exit_code1 = p1.Join()
    self.assertEqual(exit_code0, 0)
    self.assertEqual(exit_code1, 0)

  @test_util.deprecated_graph_mode_only
  def testOnDemandCompilation(self):
    if ipu.utils.running_on_ipu_model():
      self.skipTest(
          "There is no device contention with the model: nothing to test.")

    def BuildAndRunModelOnDemand(first):
      connection_type = DeviceConnectionType.ON_DEMAND
      with session.Session() as sess:
        train, loss, inp, bias = _MyNet()
        if first:
          _ConfigureSystem(connection_type)
          train = ipu.ipu_compiler.compile(lambda: (loss, train), [])
          sess.run(variables.global_variables_initializer())
          fd = {
              inp: np.random.random_sample([1] + [24] * ndims + [M * K]),
              bias: np.random.random_sample([N * K])
          }
          sess.run(train, fd)
        else:
          time.sleep(1)  # Make sure the first process goes first.
          _ConfigureSystem(connection_type)
          # Compilation should succeed as it doesn't require the device.
          train = ipu.ipu_compiler.compile(lambda: (loss, train), [])
          with self.assertRaisesRegex(Exception, "Could not attach"):
            sess.run(variables.global_variables_initializer())

    p0 = Process(BuildAndRunModelOnDemand, True)
    p1 = Process(BuildAndRunModelOnDemand, False)
    exit_code0 = p0.Join()
    exit_code1 = p1.Join()
    self.assertEqual(exit_code0, 0)
    self.assertEqual(exit_code1, 0)


if __name__ == "__main__":
  googletest.main()
