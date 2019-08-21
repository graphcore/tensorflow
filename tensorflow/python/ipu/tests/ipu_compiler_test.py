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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import googletest


class IpuCompilerTest(test_util.TensorFlowTestCase):
  def testCompileWrongDeviceRaisesException(self):
    def my_net(a):
      return a + a

    with ops.device("/device:CPU:0"):
      a = array_ops.placeholder(np.float32, shape=[1])
      with self.assertRaisesRegexp(Exception, "not placed on an IPU"):
        ipu_compiler.compile(my_net, inputs=[a])

  def testCompileNoopOnWrongDeviceIsOK(self):
    def my_net():
      return control_flow_ops.no_op()

    with ops.device("/device:CPU:0"):
      ipu_compiler.compile(my_net)

  def testCompileForDevicesInWorkerTask(self):
    def my_net(a):
      return a + a

    with ops.device("/job:worker/task:0"):
      with ops.device("/device:IPU:0"):
        a = array_ops.placeholder(np.float32, shape=[1])
        ipu_compiler.compile(my_net, inputs=[a])

      with ops.device("/device:CPU:0"):
        a = array_ops.placeholder(np.float32, shape=[1])
        with self.assertRaisesRegexp(Exception, "not placed on an IPU"):
          ipu_compiler.compile(my_net, inputs=[a])


if __name__ == "__main__":
  googletest.main()
