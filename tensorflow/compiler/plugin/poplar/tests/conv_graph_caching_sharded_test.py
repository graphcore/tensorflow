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
from test_utils import ReportJSON

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables


class ConvGraphCachingTest(xla_test.XLATestCase):
  def testConvolutionsDontMatchDifferentDevices(self):
    with self.session() as sess:
      with ops.device("/device:IPU:0"):
        x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

        with variable_scope.variable_scope("vs", use_resource=True):
          with ipu.scopes.ipu_shard(0):
            y = layers.Conv2D(
                2,
                1,
                use_bias=False,
                kernel_initializer=init_ops.ones_initializer())(x)
          with ipu.scopes.ipu_shard(1):
            y = layers.Conv2D(
                2,
                1,
                use_bias=False,
                kernel_initializer=init_ops.ones_initializer())(y)

      report = ReportJSON(self, sess, sharded=True)

      sess.run(variables.global_variables_initializer())

      report.reset()

      sess.run(y, {x: np.zeros([1, 4, 4, 2])})

      report.parse_log()

      # Note how there are two convolutions
      ok = [
          '__seed*', '*OnTileCopy*', 'vs/conv2d/Conv2D/convolution.*',
          'Copy_vs/conv2d/Conv2D/convolution.*',
          'vs/conv2d_1/Conv2D/convolution.*'
      ]
      report.assert_all_compute_sets_and_list(ok)


if __name__ == "__main__":
  googletest.main()
