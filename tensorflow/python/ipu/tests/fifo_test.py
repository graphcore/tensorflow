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


import os
import numpy as np

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python import ipu


class FifoTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testFifoCompleteLoop(self):
    def my_net(x):
      body = lambda z: ipu.internal_ops.fifo(z, 5)
      return ipu.loops.repeat(6, body, [x])

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      x = array_ops.placeholder(np.float32, shape=[2])
      run_loop = ipu.ipu_compiler.compile(my_net, inputs=[x])

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      res = sess.run(run_loop, {x: np.ones([2])})
      self.assertAllClose(res, np.ones([1, 2]))

  @test_util.deprecated_graph_mode_only
  def testFifo(self):
    def my_net(x):
      body = lambda z: ipu.internal_ops.fifo(z, 5)
      return ipu.loops.repeat(3, body, [x])

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      x = array_ops.placeholder(np.float32, shape=[2])
      run_loop = ipu.ipu_compiler.compile(my_net, inputs=[x])

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      res = sess.run(run_loop, {x: np.ones([2])})
      self.assertAllClose(res, np.zeros([1, 2]))


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
