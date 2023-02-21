# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

import os

from tensorflow.compiler.plugin.poplar.driver.config_pb2 import IpuOptions
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ipu import config
from tensorflow.python.ipu.utils import DeviceConnectionType
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class UseOnDemandTest(test_util.TensorFlowTestCase):
  def _configure(self, configured_type, expected_type):
    cfg = config.IPUConfig()
    cfg.device_connection.type = configured_type
    cfg.configure_ipu_system()

    # Get the current config.
    g = ops.Graph()
    with g.as_default():
      with ops.device("CPU"):
        with session_lib.Session(graph=g) as s:
          configurations = s.run(gen_ipu_ops.ipu_get_configuration())

    self.assertEqual(len(configurations), 1)
    actual_cfg = IpuOptions()
    actual_cfg.ParseFromString(configurations[0])

    self.assertEqual(actual_cfg.device_connection_type, expected_type)

  @test_util.deprecated_graph_mode_only
  def testNoFlag(self):
    self._configure(DeviceConnectionType.ALWAYS,
                    DeviceConnectionType.ALWAYS.value)

  @test_util.deprecated_graph_mode_only
  def testFlag(self):
    flags = os.environ.get("TF_POPLAR_FLAGS", "") + ' --use_on_demand'
    with test.mock.patch.dict("os.environ", {"TF_POPLAR_FLAGS": flags}):
      self._configure(DeviceConnectionType.ALWAYS,
                      DeviceConnectionType.ON_DEMAND.value)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = ('--tf_xla_min_cluster_size=1' +
                                os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
