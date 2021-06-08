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
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.plugin.poplar.tests.test_utils import ReportJSON
from tensorflow.python import ipu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables

dataType = np.float16

batch_size = 1
num_input = 28
timesteps = 5
num_hidden = 512


def _PopnnGRU(x, initial_state):
  gru_cell = ipu.ops.rnn_ops.PopnnGRU(
      num_hidden,
      dtype=dataType,
      weights_initializer=init_ops.zeros_initializer(dtype=dataType),
      bias_initializer=init_ops.constant_initializer(2.0, dtype=dataType))
  return gru_cell(x, initial_state=initial_state, training=False)


def _tfGRU(x, initial_state):
  gru_cell = rnn_cell.GRUCell(
      num_hidden,
      name='gru_cell',
      kernel_initializer=init_ops.zeros_initializer(dtype=dataType),
      bias_initializer=init_ops.constant_initializer(2.0, dtype=dataType))
  return rnn.dynamic_rnn(gru_cell,
                         x,
                         dtype=dataType,
                         initial_state=initial_state,
                         time_major=True)


class GRUSizeTest(xla_test.XLATestCase):
  def RunLayer(self, layer_func, x):
    with self.session() as sess:
      with ops.device('cpu'):
        px = array_ops.placeholder(dataType, shape=x.shape)
        pinitial_state = array_ops.placeholder(dataType,
                                               shape=[batch_size, num_hidden])
      with ipu.scopes.ipu_scope("/device:IPU:0"):
        r = ipu.ipu_compiler.compile(layer_func, inputs=[px, pinitial_state])

      report = ReportJSON(self, sess)

      sess.run(variables.global_variables_initializer())
      report.reset()
      result = sess.run(r, {
          px: x,
          pinitial_state: np.ones(pinitial_state.shape),
      })
      report.parse_log()
      size = report.get_total_tile_memory()
    return (size, result)

  # Test which verifies that:
  # 1. Custom op uses less memory
  # 2. Custom op and Tf op return the same result
  def testCustomOpIsSmaller(self):
    np.random.seed(42)
    x = np.random.rand(timesteps, batch_size, num_input).astype(dataType)
    size_custom_op, result_custom_op = self.RunLayer(_PopnnGRU, x)
    size_tf, result_tf = self.RunLayer(_tfGRU, x)
    self.assertTrue(size_custom_op < size_tf)
    self.assertAllClose(result_custom_op, result_tf)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
