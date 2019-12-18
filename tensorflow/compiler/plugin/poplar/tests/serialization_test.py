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

import tempfile
import os
import shutil
import numpy as np
import test_utils as tu

from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ipu import utils
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.training import gradient_descent


def numFilesInFolder(folder):
  return len([
      name for name in os.listdir(folder)
      if os.path.isfile(os.path.join(folder, name))
  ])


class IpuSerializationTest(xla_test.XLATestCase):
  def testGroupedConvolutionsSerialization(self):
    if utils.running_on_ipu_model():
      self.skipTest(
          "Serialisation of executables is only supported for IPU targets")
    ndims = 2
    M = 3
    N = 5
    K = 7  # input features per group, output features per group, number of groups

    with self.session() as sess:
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

      ipu.ipu_compiler.compile(lambda: (loss, train), [])

      with tempfile.TemporaryDirectory() as tmp:
        folder = os.path.join(tmp, "saved")
        if os.path.isdir(folder):
          shutil.rmtree(folder)

        tu.ReportJSON(self, sess, serialization_folder=folder)

        sess.run(variables.global_variables_initializer())

        self.assertTrue(os.path.isdir(folder))
        self.assertEqual(numFilesInFolder(folder), 2)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
