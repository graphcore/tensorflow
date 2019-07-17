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

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python import ipu
import os

import tensorflow as tf

class UserProvidedOpsTest(test_util.TensorFlowTestCase):
  def testUserOp(self):

    cwd = os.getcwd()
    outputs = {
      "output_types" : [tf.dtypes.float32, tf.dtypes.float32,  tf.dtypes.float32],
      "output_shapes" : [tf.TensorShape([20]), tf.TensorShape([5,2]), tf.TensorShape([10])],
    }
    lib_path = cwd + "/../../libadd_incrementing_custom.so"
    op_name = "AddIncrCustom"
    def my_net(x,y,z):
      x = ipu.internal_ops.precompiled_user_op([x,y,z], op_name, lib_path, outs=outputs)
      return x

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      x = array_ops.placeholder(np.float32, shape=[20])
      y = array_ops.placeholder(np.float32, shape=[5,2])
      z = array_ops.placeholder(np.float32, shape=[10])

      model = ipu.ipu_compiler.compile(my_net, inputs=[x, y, z])

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      res = sess.run(model, {x: np.ones([20]), y: np.ones([5,2]), z : np.ones([10])})

      self.assertAllEqual(np.full([20], 2.0), res[0])
      self.assertAllEqual(np.full([5,2], 3.0), res[1])
      self.assertAllEqual(np.full([10], 4.0), res[2])



if __name__ == "__main__":
  googletest.main()


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = (
      '--tf_xla_min_cluster_size=1 ' + os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
