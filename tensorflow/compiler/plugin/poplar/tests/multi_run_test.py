# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.ipu import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python import ipu


class IpuXlaMultiRunTest(xla_test.XLATestCase):
  @tu.skip_on_hw
  def testSimpleTwice(self):
    with ops.device("/device:IPU:0"):
      with self.session() as sess:
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        output = pa + pb

        fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[1., 2.], [6., 8.]])

        fd = {pa: [[0., 0.], [1., 1.]], pb: [[2., 1.], [4., 5.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[2., 1.], [5., 6.]])

  @tu.skip_on_hw
  def testSimpleThree(self):
    with ops.device("/device:IPU:0"):
      with self.session() as sess:
        pa = array_ops.placeholder(np.float32, [2, 2], name="a")
        pb = array_ops.placeholder(np.float32, [2, 2], name="b")
        output = pa + pb

        fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[1., 2.], [6., 8.]])

        fd = {pa: [[0., 0.], [1., 1.]], pb: [[2., 1.], [4., 5.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[2., 1.], [5., 6.]])

        fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[1., 2.], [6., 8.]])

  @tu.test_may_use_ipus_or_model(num_ipus=1)
  def testCatchException(self):
    with self.session() as sess:
      cfg = ipu.config.IPUConfig()
      cfg.auto_select_ipus = 1
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()

      def my_net(x):
        with variable_scope.variable_scope('vs', use_resource=True):
          v = variable_scope.get_variable('v', initializer=1.0)
          v = v.assign_add(1.0)
          b = control_flow_ops.Assert(x > 0.0, [x])
        with ops.control_dependencies([v, b]):
          return x + v

      with ops.device('cpu'):
        x = array_ops.placeholder(np.float32)

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        res = ipu.ipu_compiler.compile(my_net, inputs=[x])

      tu.move_variable_initialization_to_cpu()
      sess.run(variables.global_variables_initializer())

      raised = False
      try:
        sess.run(res, {x: -1.0})
      except:  # pylint: disable=bare-except
        raised = True
      self.assertTrue(raised)
      self.assertAllClose(sess.run(res, {x: 1.0}), [3.0])


if __name__ == "__main__":
  googletest.main()
