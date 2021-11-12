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
import pva

from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.utils import DeviceConnectionType
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables


class ReplicationNormaliseTest(test_util.TensorFlowTestCase):
  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testReplicationNormalise(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.auto_select_ipus = 2
    cfg.device_connection.enable_remote_buffers = True
    cfg.device_connection.type = DeviceConnectionType.ON_DEMAND
    cfg.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      y = gen_poputil_ops.ipu_replication_normalise(x)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      res = sess.run(y, {x: np.ones([1, 4, 4, 2])})
      self.assertAllClose(res, np.full([1, 4, 4, 2], 0.5))

    report = pva.openReport(report_helper.find_report())
    # pylint: disable=line-too-long
    ok = [
        'setStochasticRounding',
        '__seed',
        'IpuReplicationNormalise/replication-normalise*/replication_normalise/Op/Mul',
    ]
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testReplicationNormaliseNotInplace(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = False
    cfg.auto_select_ipus = 2
    cfg.device_connection.enable_remote_buffers = True
    cfg.device_connection.type = DeviceConnectionType.ON_DEMAND
    cfg.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      a = gen_poputil_ops.ipu_replication_normalise(x)
      b = a + x

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      res = sess.run(b, {x: np.ones([1, 4, 4, 2])})
      self.assertAllClose(res, np.full([1, 4, 4, 2], 1.5))

    report = pva.openReport(report_helper.find_report())
    # pylint: disable=line-too-long
    ok = [
        'setStochasticRounding',
        '__seed*',
        'IpuReplicationNormalise/replication-normalise*/replication_normalise/Op/Mul',
        'add/add*/Add',
        '[cC]opy_',
    ]
    # pylint: enable=line-too-long
    self.assert_all_compute_sets_and_list(report, ok)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1 ' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
