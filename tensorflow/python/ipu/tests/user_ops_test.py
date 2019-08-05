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
import os

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class UserProvidedOpsTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testUserOp(self):

    cwd = os.getcwd()
    outputs = {
        "output_types": [dtypes.float32, dtypes.float32, dtypes.float32],
        "output_shapes": [
            tensor_shape.TensorShape([20]),
            tensor_shape.TensorShape([5, 2]),
            tensor_shape.TensorShape([10])
        ],
    }
    lib_path = cwd + "/tensorflow/python/ipu/libadd_incrementing_custom.so"

    def my_net(x, y, z):
      x = ipu.internal_ops.precompiled_user_op([x, y, z],
                                               lib_path,
                                               outs=outputs)
      return x

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      x = array_ops.placeholder(np.float32, shape=[20])
      y = array_ops.placeholder(np.float32, shape=[5, 2])
      z = array_ops.placeholder(np.float32, shape=[10])

      model = ipu.ipu_compiler.compile(my_net, inputs=[x, y, z])

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      res = sess.run(model, {
          x: np.ones([20]),
          y: np.ones([5, 2]),
          z: np.ones([10])
      })

      self.assertAllEqual(np.full([20], 2.0), res[0])
      self.assertAllEqual(np.full([5, 2], 3.0), res[1])
      self.assertAllEqual(np.full([10], 4.0), res[2])


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = (
      '--tf_xla_min_cluster_size=1 ' + os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
