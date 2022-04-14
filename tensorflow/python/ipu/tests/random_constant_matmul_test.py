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

from tensorflow.python.ipu import test_utils as tu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python import ipu


class RandomConstantTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testRandomConstant(self):
    def my_net(x, w):
      b = random_ops.random_uniform([2, 2])
      return math_ops.matmul(x, w) + b

    with ipu.scopes.ipu_scope('/device:IPU:0'):
      x = array_ops.placeholder(np.float32, shape=[2, 3])
      w = array_ops.placeholder(np.float32, shape=[3, 2])
      run_loop = ipu.ipu_compiler.compile(my_net, inputs=[x, w])

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      # We don't care about the value, just that it doesn't throw an exception
      sess.run(run_loop, {x: np.ones([2, 3]), w: np.ones([3, 2])})


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
