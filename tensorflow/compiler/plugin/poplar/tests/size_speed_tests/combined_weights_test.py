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

import numpy as np
import pva

import tensorflow.compiler.plugin.poplar.tests.test_utils as tu
from tensorflow.compiler.tests import xla_test
from tensorflow.python import ipu
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest

# This verifies that in a replicated graph, the weights are copied to IPU 0 on
# the device as a merged set, then copied to the other IPUs as a merged set.

datatype = np.float16


def _get_variable(name, shape, init):
  return vs.get_variable(name, shape, initializer=init, dtype=datatype)


def inference(x):

  with vs.variable_scope('all', use_resource=True):
    x = fc("fc0", x, 256)
    x = fc("fc1", x, 256)
    x = fc("fc2", x, 256)
    x = fc("fc3", x, 256)
    x = fc("fc4", x, 256)
    x = fc("fc5", x, 256)
    x = fc("fc6", x, 256)
    x = fc("fc7", x, 256)
    x = fc("fc8", x, 256)
    x = fc("fc9", x, 256)

  return x


def fc(name, x, num_units_out):
  num_units_in = x.shape[1]
  weights_initializer = init_ops.truncated_normal_initializer(stddev=0.01)

  with vs.variable_scope(name):
    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            init=weights_initializer)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           init=init_ops.constant_initializer(0.0))

    x = nn_ops.xw_plus_b(x, weights, biases)

  return x


class CombinedWeightsTest(xla_test.XLATestCase):
  def testMergedWeightDownload(self):
    cfg = ipu.utils.IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.auto_select_ipus = 1
    cfg.ipu_model.compile_ipu_code = True
    cfg.configure_ipu_system()

    with self.session() as sess:
      x = array_ops.placeholder(datatype, shape=[16, 4])
      y_ = array_ops.placeholder(datatype, shape=[16, 256])

      with ipu.scopes.ipu_scope("/device:IPU:0"):
        logits = inference(x)

        loss = math_ops.reduce_mean(
            nn_ops.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=array_ops.stop_gradient(y_)))

      sess.run(variables.global_variables_initializer())
      report_helper.clear_reports()

      data = np.zeros([16, 4])
      labels = np.zeros([16, 256])

      sess.run(loss, feed_dict={x: data, y_: labels})

    report = pva.openReport(report_helper.find_report())

    # Find the first case - the download weights sequence
    download_weights = next(
        t for t in report.compilation.programs
        if t.type == pva.Program.Type.OnEveryTileSwitch).children[0]

    self.assertLess(
        len(download_weights.children), 12,
        "The download weights sequence should not have lots of entries "
        "(because the copies will have been merged)")

    # Also check the overall size
    self.assert_total_tile_memory(report, 1559766, tolerance=0.1)


if __name__ == "__main__":
  googletest.main()
