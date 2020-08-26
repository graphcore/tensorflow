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

from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables


class ReplicationNormaliseTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testReplicationNormalise(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      y = gen_poputil_ops.ipu_replication_normalise(x)

    with tu.ipu_session() as sess:
      report = tu.ReportJSON(self, sess, replicated=True)
      sess.run(variables.global_variables_initializer())

      report.reset()

      res = sess.run(y, {x: np.ones([1, 4, 4, 2])})
      self.assertAllClose(res, np.full([1, 4, 4, 2], 0.5))
      report.parse_log()

      ok = [
          '__seed*',
          'IpuReplicationNormalise/replication-normalise*/replication_normalise/Op/Divide',
          'switchControlBroadcast*/GlobalPre/Copy/OnTileCopy',
          '/OnTileCopy',
          'Copy_XLA_Args*OnTileCopy',
      ]
      report.assert_all_compute_sets_and_list(ok)

  def testReplicationNormaliseNotInplace(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      a = gen_poputil_ops.ipu_replication_normalise(x)
      b = a + x

    with tu.ipu_session() as sess:
      report = tu.ReportJSON(self, sess, replicated=True)
      sess.run(variables.global_variables_initializer())

      report.reset()

      res = sess.run(b, {x: np.ones([1, 4, 4, 2])})
      self.assertAllClose(res, np.full([1, 4, 4, 2], 1.5))
      report.parse_log()

      ok = [
          '__seed*',
          'IpuReplicationNormalise/replication-normalise*/replication_normalise/Op/Divide',
          'switchControlBroadcast*/GlobalPre/Copy/OnTileCopy',
          '/OnTileCopy',
          'Copy_XLA_Args*OnTileCopy',
          'add/add*/AddTo',
      ]
      report.assert_all_compute_sets_and_list(ok)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
